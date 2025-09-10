# SAS — Iterative HW–SW Co-Design: Memory-Aware Layout Re-Optimization

**한 줄 요약**
`permute → transform(S/Q/K) → re-permute` 루프를 라이브러리/CLI 한 벌로 제공한다. 핵심은 공접근 가중치 $W$ 생성–모듈러리티 기반 재퍼뮤테이션–측정(NCU/NVML)–리포트 자동화. 실패 시 즉시 롤백.

---

## 0) 가정·제약(Assumptions & Constraints)

* **환경**: Python 3.10, CUDA 11.x+/MPS/CPU 가드. Nsight Compute(ncu), NVML 전력 읽기 가능. A100/H100 1대 이상.
* **범위**: 추론(inference) 경로. 학습/분산 스케줄링/클러스터 오케스트레이션 불포함.
* **목표(요약)**: 대표 2 워크로드(SSM-Mamba-3B, Transformer-BERT-base)에서 Latency −20%, L2 hit +10%p, EpT −15% 달성(MVP).

---

## 1) 시스템 목표·논외(Scope)

* **In-scope**

  * $W$ 생성(실추적/모의), 재퍼뮤테이션 솔버(스펙트럴+로컬), S/Q/K 변환 어댑터, 파이프라인 러너, 계측/리포트, TVM/StableHLO 브리지 **PoC**.
* **Out-of-scope**

  * 모델 학습 파이프라인, 새 하드웨어 설계, 분산 추론, 상용 클러스터 운영 자동화.

---

## 2) 상위 아키텍처(Overview)

```
┌────────────────────────────────────────────────────────────────┐
│                         icd CLI / Python API                   │
├────────────────────────────────────────────────────────────────┤
│                        runtime/orchestrator                    │
│  (파이프라인 스케줄·재시도·롤백·아티팩트 경로·캐시 관리)          │
├───────────────┬─────────────────────┬─────────────────┬─────────┤
│ core/graph    │ core/solver         │ adapters/       │ measure │
│ (trace→W)     │ (spectral+local)    │  S | Q | K      │ (ncu,nvml,
│               │                     │ (변환/메타데이터)│ timers, reporter)
├───────────────┴───────────────┬─────┴─────────────────┴─────────┤
│ bridge/ (TVM or StableHLO PoC)│     storage/ (perm/W/log/art)   │
└───────────────────────────────┴─────────────────────────────────┘
```

* **core/graph**: 공접근 가중치 행렬 $W$ 생성(실행 추적 or 모의).
* **core/solver**: 스펙트럴 정렬(Fiedler) + 로컬 탐색(2-opt/adj-swap). 비용 $C(\pi)$/모듈러리티 $Q$ 계산.
* **adapters/**: 상태-변환 3종(S: sparsity(2:4/HDS & unstructured), Q: PTQ/FP8 스텁, K: KV 캐시 압축). 변환 후 메타데이터 갱신·트리거.
* **runtime/**: 파이프라인 오케스트레이터(컨트롤 플로, 캐시, 에러·롤백, 재시도, 시드/클럭 고정).
* **measure/**: Nsight Compute, NVML, 벽시계 타이머 래퍼. HTML/CSV 리포터.
* **bridge/**: IR(StableHLO/TVM) 상 삽입 지점과 변환·메타데이터 패스 **PoC**.
* **storage/**: 아티팩트 디렉토리(perm.json, W\.npz, ncu.rep, power.csv, report.html).

---

## 3) 데이터 모델(Data & Artifacts)

* **W(n×n)**: float32, 대칭 희소 행렬(CSR/COO). 스키마: `{"shape": n, "format": "csr", "nnz": m, "source": "trace|mock", "seed": int}`
* **π (Permutation)**: int32 배열 길이 n. 버전태그: hash(model+task+S/Q/K+seed).
* **Transform Meta**: `{ sparsity: {type: "2:4|unstructured", rate}, quant: {dtype: "int8|fp8", method}, kv: {block: int, drop: float} }`
* **Metrics Record**: `{lat_ms, toks_per_s, l2_hit_pct, ept_j_per_tok, hw, driver, seed, clock cap, repeats}`
* **Reports**: `report.html|.csv`(전/후 비교, 표·그림), `ncu.json`, `power.csv`.

---

## 4) 컨트롤 플로(파이프라인)

### 기본 시퀀스(Linear vs Iterative)

1. **BuildW**: 실행 추적 or 모의로 $W$ 생성/정규화.
2. **Permute**: `π₀ ← solver.fit(W)`; 레이아웃 적용.
3. **Transform**: S/Q/K 중 하나 이상 적용(메타 갱신).
4. **Re-permute**: `π₁ ← solver.fit(W’)`; 상태 변화 반영.
5. **Run/Measure**: 동일 입력 배치로 `lat/l2/EpT` 측정.
6. **Report/Cache**: 전/후 비교, `π₁` 캐시. 실패 시 `π₀`로 롤백.

> **Baseline**은 4단계(Re-permute) 생략.

### 상태머신(State)

* `READY → W_BUILT → PERMUTED → TRANSFORMED → REPERMUTED → MEASURED → REPORTED`
* 실패 시 `→ ROLLBACK(π_prev) → REPORTED`.

---

## 5) 컴포넌트 명세(Responsibilities & Contracts)

### 5.1 core/graph

* 입력: 실행 추적(trace events) 또는 모의 설정(seed, blocks, noise).
* 출력: $W$ (CSR/COO).
* 계약: 동일 입력→결정론적 $W$. `nnz/shape` 상한 검사, NaN 금지.

### 5.2 core/solver

* 입력: $W$, 시간상한(soft), 로컬탐색 단계수, k(블록 수, 선택).
* 출력: permutation `π`, 보조지표 `C(π), Q(π)`.
* 계약: 시간상한 내 최적화 종료. 개선 실패 시 초기해 반환(플래그 표시).

### 5.3 adapters/S|Q|K

* 입력: 텐서·레이아웃 메타.
* 출력: 변경된 텐서/가중치와 **변환 메타데이터**.
* 계약: 손실/정밀도 영향 범위 보고, 재퍼뮤테이션 트리거 조건 반환. 실패 시 no-op.

### 5.4 runtime/orchestrator

* 기능: 파이프라인 스케줄, 재시도/백오프, 실패 롤백, 캐시(hit/miss), 시드/클럭 고정, 아티팩트 경로 관리.
* 계약: 모든 스텝의 로깅/이벤트 발행(오버헤드 <1%).

### 5.5 measure/

* `ncu`: L2 hit 수집, 프로파일 키 세트 선택.
* `nvml`: 전력 샘플링(주기), EpT 계산.
* 계약: 워밍업 제외, 반복 횟수 N, 고정클럭/전력캡 옵션 준수.

### 5.6 bridge/

* StableHLO or TVM Pass 파이프라인에 **삽입 지점** 정의(예: layout-tag attach → pass → lower).
* 계약: PoC 단계는 **메타데이터 왕복**과 최소 변환만 수행.

---

## 6) 인터페이스(ICD 요약·시그니처)

```python
# graph
W = build_w(source: Literal["trace","mock"], **cfg) -> csr_matrix

# solver
pi, stats = fit_permutation(W, time_budget_s=300, refine_steps=2_000)  # stats: {C, Q}

# adapters
out, meta = apply_sparsity(tensor, type="2:4", rate=0.5)
out, meta = apply_quant(tensor, dtype="int8", method="ptq-minmax")
out, meta = apply_kvcache(cache, block=128, drop=0.1)

# orchestrator (CLI 내부)
run(config: Dict) -> RunArtifacts  # artifacts paths, metrics summary
```

(세부 타입/예외/에러코드는 ICD 문서에 확정. 여기서는 축약.)

---

## 7) 성능·리소스 예산(Budgets)

* **Re-permute 오버헤드**: D≈2.5k 기준 ≤ 5분(오프라인). 대규모는 샘플링/부분고유값/블록 병렬.
* **메모리 상한**: $W$ CSR nnz ≤ 0.05·D²(모의), 실추적은 nnz≤ trace-bound.
* **계측 오버헤드**: ncu/NVML 포함 **< 5%**(프로파일 샘플링 비율 조절).
* **보고 시간**: HTML/CSV 생성 < 30초.

---

## 8) 동시성·실행모델(Concurrency)

* 파이프라인 단계는 **동기**(determinism 우선).
* 내부 병렬: 스펙트럴(부분 고유값)·로컬탐색(스레드-샘플) **데이터 병렬** 허용.
* 계측은 **분리 프로세스**(ncu CLI)로 격리. 파일락으로 아티팩트 충돌 방지.

---

## 9) 관측성/로깅(Observability)

* **이벤트 스키마**: `stage, t_start, t_end, ok, meta{…}`
* **카운터**: `lat_ms, toks_s, l2_hit_pct, ept, Q, C, nnz(W)`
* **샘플링**: ncu는 샘플/풀 프로파일 구분.
* **오버헤드 가드**: 관측성 자체 오버헤드 측정·보고.

---

## 10) 오류·장애 모델(Error Model) & 롤백

* **그래프 실패**: trace 손상/빈도 0 → mock 대체 or 중단.
* **솔버 타임아웃**: 초기해 반환(플래그) → 선택적 재시도.
* **어댑터 실패**: no-op + 경고, 파이프라인 지속.
* **계측 실패**: 대체 지표(벽시계)로 다운그레이드.
* **품질 저하 탐지**: Latency/L2/EpT 악화 시 자동 롤백(직전 π).

---

## 11) 보안·라이선스·컴플라이언스

* **SBOM** 자동 생성, 라이선스 스캔(CI).
* 데이터셋/모델 카드 준수, 로그에 민감정보 저장 금지.
* 프로파일 결과는 로컬 보관 기본, 업로드는 opt-in.

---

## 12) 배포·환경(Deployment)

* 배포: PyPI 패키지 + `icd` CLI. `pip install icd-co`
* **Dockerfile** 제공(드라이버 매칭 제외), conda env export 포함.
* 장치 가드:

  ```python
  import torch
  device = ("mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu")
  ```

  (PyTorch 사용 경로에 한정. TVM/StableHLO 경로는 별도 가드.)

---

## 13) 테스트·CI(Testing Architecture)

* **유닛**: solver 단조성(블록 강도↑ → C↓, Q↑), 그래프 결정론, 어댑터 no-op 안전성.
* **통합**: Linear vs Iterative 전/후 지표 동시 출력.
* **E2E**: 대표 2 워크로드 자동 재현.
* **회귀**: 마이크로벤치(지연/토큰/s) 허용 편차 게이트.
* **결정론**: seed/클럭 고정, 반복 N≥30, CI에서 분산 95% CI 체크.

---

## 14) 설계 선택(Design Decisions) — 요지

* **모듈러리티 기반**: 캐시라인(블록) 현실 반영 → TSP-류 쌍최적화보다 목적함수 정합성 높음.
* **스펙트럴+로컬**: 해석가능·안정적 초기해 + 경량 개선. 학습형 스케줄러는 2단계로 추후 옵션.
* **IR 브리지 PoC**: 초기엔 메타 태그 왕복·최소 pass만. 본격 통합은 2단계 로드맵.

---

## 15) 리스크·완화

* **계측 변동성**: 드라이버/온도/클럭 → SOP(고정클럭·워밍업·반복) 강제, 로그에 환경 해시.
* **스케일**: D↑ 시 O(D³) → 랜치/부분고유값, 블록 독립 최적화.
* **일반화**: 2 워크로드 편향 → Ablation 스윕(희소율/정밀도/길이), 추가 모델 1종 옵션.
* **통합비용**: 프레임워크 다양성 → CLI 우선, IR 통합은 opt-in.

---

## 16) 수용 기준(Acceptance)

* 대표 2 워크로드에서 **Latency −20% / L2 +10%p / EpT −15%** 동시 달성(품질 저하 ≤ 0.1%p).
* 실패 시 롤백·리포트·로그가 자동 생성되고 원인 역추적 가능.
* 외부 검증자 기준 **24h 내** 재현 패키지 성공.

---

## 17) 실행 체크리스트(Owner 보기)

* [ ] SOP(고정클럭/워밍업/반복 N) 확정
* [ ] ICD/Pass 포인트 초안 확정(브리지 PoC)
* [ ] $W$ 스키마·압축 포맷 결재
* [ ] 솔버 시간상한/스케일 전략 선택
* [ ] 관측성 스키마/오버헤드 측정 케이스 통과
* [ ] 회귀 게이트(threshold) CI에 적용

---

## 18) 구성 키(Config Keys) — 발췌

```yaml
pipeline:
  mode: iterative   # linear|iterative
  repeats: 1000
  fixed_clock: true
  warmup_iter: 50
graph:
  source: trace     # trace|mock
  mock:
    blocks: 4
    noise: 0.02
    seed: 42
solver:
  time_budget_s: 300
  refine_steps: 2000
  k_blocks: 4
transform:
  sparsity: {type: "2:4", rate: 0.5}
  quant:    {dtype: "int8", method: "ptq-minmax"}
  kv:       {block: 128, drop: 0.10}
measure:
  ncu_metrics: ["l2_tex__t_sector_hit_rate.pct"]
  power_sample_hz: 10
report:
  out_dir: "runs/exp001"
```

---

### 부록 — 간단 데이터플로(DFD)

`trace/mock → W(CSR) → π₀ → S/Q/K → W' → π₁ → run → {lat, l2, ept} → report.html`

---
