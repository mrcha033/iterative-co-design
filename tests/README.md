# 테스트 플랜 — Iterative HW–SW Co-Design (Layout Re-Optimization)

**한 줄 요약**
코어 알고리즘·IR 패스·측정 파이프라인을 단위→통합→E2E→성능/회귀→결정론 순으로 검증하고, PRD 지표(Latency−20%, L2+10%p, EpT−15%)를 게이트로 삼아 CI에 자동화한다.

---

## 0) 가정·제약

* 언어/런타임: Python 3.10, pytest.
* 장치: A100/H100(권장). Nsight Compute(ncu), NVML(또는 대체 전력계).
* 범위: 추론(inference) 경로. 학습/분산 제외.
* 소스 구조: `core/`, `adapters/`, `runtime/`, `measure/`, `bridge/`, `tests/`, `runs/`.

---

## 1) 목표/범위

* **정확성**: 모듈러리티 기반 재퍼뮤테이션이 비용 $C$↓, $Q$↑를 일관되게 달성.
* **합법성**: IR 패스 적용 후 shape/dtype/stride/align 위반 0.
* **효율**: 대표 2 워크로드에서 PRD 목표 달성.
* **재현성/결정론**: 동일 seed/클럭에서 변동률 ≤ 1% (핵심 지표).
* **관측성**: 메트릭/로그/아티팩트 완전 수집.

---

## 2) 테스트 계층/매트릭스

| 계층               | 목적                   | 대표 스위트                                | 트리거/빈도  | 게이트       |
| ---------------- | -------------------- | ------------------------------------- | ------- | --------- |
| Unit             | 함수 단위 정합/수치 안정       | `tests/unit/test_cost.py`             | PR/커밋마다 | 필수 통과     |
| Integration      | 모듈간 계약/데이터 흐름        | `tests/integration/test_pipeline.py`  | PR/일일   | 필수 통과     |
| IR Pass          | StableHLO/TVM 변환 합법성 | `tests/ir/test_pass_filecheck.py`     | PR/주간   | 필수 통과     |
| E2E(Mock)        | 모의 $W$에서 메커니즘 검증     | `tests/e2e/test_mock_iterative.py`    | PR/일일   | 필수 통과     |
| Perf/Regression  | Lat/L2/EpT 회귀 방지     | `tests/perf/test_benchmarks.py`       | 야간/주간   | 게이트(허용편차) |
| Determinism      | seed/클럭 고정 시 동일 결과   | `tests/stability/test_determinism.py` | 야간      | 필수 통과     |
| Failure/Fallback | ncu/NVML 등 실패 경로     | `tests/failure/test_degrade_paths.py` | 주간      | 필수 통과     |
| Repro/AE         | 24h 재현 패키지           | `scripts/repro_smoke.sh`              | 릴리즈 전   | 필수 통과     |

---

## 3) 테스트 케이스 상세

### 3.1 단위(Unit)

**U-01 비용 단조성**

* 입력: 블록 구조 강도 $ρ\in\{0.2,0.5,0.8\}$의 합성 $W$.
* 검증: `fit_permutation` 후 $\widetilde{C}\downarrow$, $\widetilde{Q}\uparrow$.
* 기준: 각 ρ에서 $\widetilde{C}$ ≥10% 감소, $\widetilde{Q}$ 증가.

**U-02 스케일 불변성**

* 입력: $W$와 $aW$(a>0).
* 검증: $\arg\min J$ 동일(π 해시 동일), 지표 순서보존.

**U-03 정렬/얼라인 정칙화**

* 입력: 다양한 `vec_width, align` 설정.
* 검증: `R_align` 감소 시 `J`도 감소(다른 항 상수).

**U-04 안정성 정칙화**

* 입력: $\pi_0$ 대비 허용 이동폭 h 테스트.
* 검증: h 축 증가 시 `R_stability` 감소, 롤백 조건 정상 동작.

**U-05 그래프 생성 결정론**

* 입력: 동일 trace/mock cfg.
* 검증: `build_w` 반환 CSR 해시 동일, NaN/음수 없음.

### 3.2 통합(Integration)

**I-01 파이프라인 전/후 메트릭 산출**

* 시나리오: `linear` vs `iterative` 동일 입력.
* 검증: 아티팩트 모두 생성(`perm*.json`, `w.csr.npz`, `metrics.json`), 로그 단계 순서·타임스탬프 유효.

**I-02 캐시/재사용**

* 시나리오: 동일 조건 2회 실행.
* 검증: 2회차 `fit_permutation` 생략(hit), 결과 동일.

**I-03 트리거/롤백**

* 시나리오: 의도적으로 개선폭 ε 미만.
* 검증: `redecide-permutation` → 롤백, 리포트에 원인 명시.

### 3.3 IR Pass

**P-01 메타태그 왕복**

* 입력: StableHLO 모듈.
* 검증: `attach → pass → lower` 후 `icd.layout_perm` 보존(FileCheck).

**P-02 합법화/정합성**

* 입력: transpose/reshape 상호작용 케이스.
* 검증: shape/dtype 보존, align 위반 0, alias 충돌 0.

**P-03 결정론/시간상한**

* 검증: 동일 seed/time\_budget → 동일 π, `elapsed_s ≤ budget`.

### 3.4 E2E(Mock)

**E-01 모의 $W$ 메커니즘**

* 입력: `make_block_matrix(D=256, blocks=4, noise=0.02)`.
* 검증: iterative가 linear 대비 $\widetilde{C}$ 10%↓, $\widetilde{Q}$↑.
* 아티팩트: `report.html` 비교 그래프(Q–Latency 상관 스케치).

**E-02 경계 조건**

* 입력: D<128, 짧은 시퀀스.
* 검증: 개선 미미/0이어도 **테스트 통과**(문서된 경계).

### 3.5 성능/회귀(Perf/Regression)

> CI에서는 **스모크 게이트**(완화 기준), 릴리즈/야간에서는 **PRD 게이트**(엄격 기준).

| 케이스           | 대상        | 기준(스모크)       | 기준(엄격)   |
| ------------- | --------- | ------------- | -------- |
| B-01 Latency  | 대표 2 워크로드 | −10% 이상       | −20% 이상  |
| B-02 L2 hit   | 대표 2 워크로드 | +5%p 이상       | +10%p 이상 |
| B-03 EpT      | 대표 2 워크로드 | −8% 이상        | −15% 이상  |
| B-04 최적화 오버헤드 | D≈2.5k    | ≤ 6분          | ≤ 5분     |
| B-05 ncu 오버헤드 | 프로파일 모드   | Lat 악화 ≤ +10% | ≤ +5%    |

* 통계: n≥3 반복, 95% CI 포함. 효과크기(Hedges g) 병기.

### 3.6 결정론/안정성(Stability)

**S-01 seed/클럭 결정론**

* 동일 seed/고정클럭: π 해시 동일, 지표 변동률 ≤ 1%.

**S-02 노이즈 내성**

* 고정클럭 해제: 반복 n≥10에서 평균±CI 변동 기록(문서화만).

### 3.7 실패/대체 경로(Fallback)

**F-01 ncu 미가용**

* 상황: ncu PATH 제거.
* 검증: `MeasureError` 처리→L2 미보고, 실행 지속.

**F-02 NVML 미가용**

* 검증: EpT=N/A 보고, 나머지 지표 정상.

**F-03 Solver Timeout**

* 검증: 초기해 반환, 플래그 기록, 파이프라인 계속.

---

## 4) 메트릭·판정 규칙

* **통과 조건**: 각 테스트의 **Acceptance Criteria** 만족.
* **회귀 게이트**: `perf_baseline.json` 대비 Δ가 허용 범위 밖이면 실패.
* **결정론 게이트**: core 지표(C,Q,J,π\_hash) 변동률 > 1% → 실패.
* **통계 표기**: 모든 성능 주장은 Δ값 + 95% CI 동시 표기.

---

## 5) 아티팩트/로그 수집

* `metrics.json`, `ncu.json`, `power.csv`, `run.log`, `config.lock.json`, `report.{html,csv}`.
* 실패 시 **RCA 템플릿** 자동 생성(환경/입력/커널/솔버/통계 체크리스트).

---

## 6) 샘플 테스트 스켈레톤(pytest)

```python
# tests/unit/test_cost.py
import numpy as np
from mock.make_block_matrix import make_blocky
from core.repermute import reorder, cache_cost, modularity

def test_cost_monotonicity():
    W = make_blocky(D=256, blocks=4, noise=0.02, seed=0)
    pi_id = np.arange(W.shape[0])
    pi = reorder(W)
    c0, c1 = cache_cost(W, pi_id), cache_cost(W, pi)
    q0, q1 = modularity(W, pi_id), modularity(W, pi)
    assert c1 <= 0.9 * c0
    assert q1 >= q0

def test_graph_determinism():
    W1 = make_blocky(D=128, seed=7)
    W2 = make_blocky(D=128, seed=7)
    assert np.allclose(W1.toarray() if hasattr(W1,"toarray") else W1,
                       W2.toarray() if hasattr(W2,"toarray") else W2)
```

```python
# tests/integration/test_pipeline.py
import json, subprocess, os

def run_icd(mode, out):
    subprocess.check_call(["icd","run","-c","tests/data/config.yaml",
                           "--override", f"pipeline.mode={mode}",
                           "--out", out])

def test_pipeline_artifacts(tmp_path):
    run_icd("linear",   tmp_path/"linear")
    run_icd("iterative",tmp_path/"iter")
    for d in ["linear","iter"]:
        p = tmp_path/d
        for f in ["metrics.json","config.lock.json","run.log"]:
            assert (p/f).exists()
    with open(tmp_path/"linear"/"metrics.json") as f: m0=json.load(f)
    with open(tmp_path/"iter"/"metrics.json") as f: m1=json.load(f)
    assert m1["latency_ms"]["mean"] < m0["latency_ms"]["mean"] * 0.9  # smoke
```

---

## 7) CI 파이프라인(요약)

**Jobs**

1. `lint`: black/ruff/mypy, schema 검증.
2. `unit`: `pytest -q tests/unit`.
3. `integration`: mock 기반 통합 스위트.
4. `ir-pass`: FileCheck/스냅샷.
5. `perf-smoke`: 소형 워크로드(짧은 반복) 스모크 게이트.
6. `artifact-upload`: `runs/*` 업로드(HTML/CSV/로그).

**실패 기준**

* 어떤 게이트라도 실패 → PR 차단.
* `perf-smoke`는 스모크 기준(B-01\~B-03)으로만 차단, 야간 `perf-full`에서 PRD 기준.

**GitHub Actions (발췌)**

```yaml
jobs:
  unit:
    runs-on: ubuntu-latest
    steps: [ ... , {run: "pytest -q tests/unit"} ]
  integration:
    needs: unit
    steps: [ ... , {run: "pytest -q tests/integration"} ]
  perf_smoke:
    needs: integration
    if: github.event_name == 'pull_request'
    steps: [ ... , {run: "pytest -q tests/perf -k smoke"} ]
```

---

## 8) 커버리지/품질 목표

* **코드 커버리지**: 라인 ≥ 85%, 브랜치 ≥ 70% (핵심 모듈 90%↑).
* **테스트 시간**: PR 스모크 ≤ 15분, 야간 전체 ≤ 2h.
* **재현성**: 외부 검증자가 `make repro`로 24h 내 전 결과 재생산.

---

## 9) 릴리즈 전 체크리스트

* [ ] Unit/Integration/IR/E2E 전부 통과
* [ ] Perf-full에서 PRD 게이트 충족(B-01\~B-03)
* [ ] Determinism 통과(±1%)
* [ ] RCA 오픈 이슈 0건(차단급)
* [ ] Repro Pack 최신화 및 실행 로그 포함

---

## 10) 역할/소유(RACI)

* **QE Owner**: 전체 플랜/CI 운영, 게이트 튜닝
* **Core Eng.**: solver/비용모델 유닛·성능
* **Systems Eng.**: runtime/measure, 결정론
* **Compiler Eng.**: IR pass FileCheck/합법화
* **PM**: PRD 지표 ↔ 게이트 정합성 확인

---

## 11) 리스크/완화

* **측정 변동성**: 고정클럭·워밍업·반복수 강화, 분리 러너.
* **장치 이질성**: 아키별 메트릭 맵 테이블 유지, 섹션 기반 ncu 사용.
* **시간 상한 초과**: 솔버 budget/샘플링·부분고유값.
* **과적합(튜닝→특정 워크로드)**: 추가 워크로드·Ablation 스윕 주기적 포함.

---

## 12) 첨부/템플릿

* `tests/data/config.yaml`(스모크 설정)
* `scripts/repro_smoke.sh`(E2E 모의 재현)
* `scripts/collect_artifacts.py`(HTML/CSV/로그 압축)
* `docs/RCA_template.md`(원인 분석 폼)

---
