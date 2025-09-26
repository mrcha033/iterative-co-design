# PRD — Iterative HW–SW Co-Design for Memory-Aware Layout Re-Optimization

**한 줄 요약**
상태 변환(희소화/양자화/KV 캐시 압축) 이후 **레이아웃 재퍼뮤테이션**으로 \*\*L2 hit↑/지연↓/EpT↓\*\*를 일관되게 달성하는 **ML-시스템** 제품(라이브러리+CLI)의 목표·범위·지표·리스크를 고정한다.

---

## 1) 배경/문제정의

* 최신 모델(SSM/Transformer)은 **I/O-bound** 구간이 커서 **데이터 이동**이 병목.
* 기존 파이프라인은 `permute → transform(S/Q/K)`처럼 **단회 최적화**에 고정되어 **상태 변화 후의 비용지형**을 반영하지 못함.
* 목표: **transform 직후 재-permute**를 자동 실행해 **캐시 지역성(=L2 hit)** 을 높이고, **지연/에너지(EpT)** 를 낮춘다.

## 2) 목표(Outcome) / 비즈니스 임팩트

* **모델 추론 효율**: 동일 품질에서 **지연 −20%**, **L2 hit +10%p**, **EpT −15%** 이상(대표 워크로드 기준).
* **개발자 생산성**: 1줄 API/CLI로 기존 파이프라인에 **비침투적** 통합.
* **재현성/채택성**: 공개 스크립트+로그로 **AE(Artifact Evaluation)** 통과 수준.

## 3) 범위(In-scope) / 논외(Out-of-scope)

**In-scope**

* 레이아웃 최적화 코어(스펙트럴 초기화+로컬 서치), S/Q/K 상태-변환 어댑터, 비용모델, 측정 파이프라인(L2/Latency/EpT), Python 라이브러리/CLI, TVM 또는 StableHLO **연동 PoC**.

**Out-of-scope**

* 학습 가속/분산 파이프라인, 커스텀 하드웨어 설계, 대형 모델 전면 학습, 상용 클러스터 오케스트레이션.

## 4) 사용자 시나리오(대표 플로우)

1. 사용자는 기존 추론 스크립트에서(현재 리포지토리 기준):

   ```bash
   python -m icd.cli.main run -c configs/mock.json --override pipeline.mode=iterative --out runs/iter
   ```
   보다 많은 예시는 docs/USAGE.md 참고.
2. 프레임워크가 `(permute → transform → re-permute)`를 실행하고, **전/후 비교 리포트**(L2 hit/Latency/EpT)를 저장.
3. 사용자는 **개입 없이** 최적 permutation을 캐시하고 재사용.

## 5) 성공 메트릭(정의/측정법)

* **Latency(ms)**: 1000회 추론 평균 벽시계 시간(워밍업 제외, 고정 클럭/전력캡 옵션).
* **Throughput(tokens/s)**: 동일 조건 하 평균 처리량.
* **L2 Hit(%)**: Nsight Compute/동등 도구의 L2 hit rate.
* **Energy-per-Token(J/token, EpT)**: NVML/전력계 샘플 적분 ÷ 생성 토큰 수.
* **채택성**: 1줄 API 통합 시간 ≤ 30분(가이드 기준), 실패율 < 5%.

> **수용 기준(Acceptance)**
>
> * 대표 2 워크로드(SSM-Mamba-3B, Transformer-BERT-base)에서 Latency −20%/L2 +10%p/EpT −15% **모두 충족**, 품질 저하 ≤ 0.1%p.
> * `icd` CLI와 Python API, 리포트 HTML/CSV 아티팩트 제공.
> * 재현 스크립트로 외부 검증자가 **24h 내** 전 결과 재생산.

## 6) 요구사항(Functional/Non-functional)

### Functional

* `repermute.fit(W|trace)` : 공접근 가중치 W(또는 트레이스)로 permutation 도출.
* `transform.apply({sparsity|quant|kv})` : 상태-변환 실행, 메타데이터 갱신.
* `icd run …` : 파이프라인 오케스트레이션(로그/프로파일/리포트).
* `collect_correlations/cluster_graph` : transform 이후 activation 기반 상관행렬 및 Louvain 클러스터로 재퍼뮤테이션 초기화.
* `measure.builtin=benchmark` : 내장 GPU 벤치마크로 Latency/Throughput/EpT 자동 취합, Nsight/NVML 연동 옵션.
* 캐시/재사용: 동일 조건에서 permutation 재사용 + 버전태깅.

### Non-functional

* **성능**: D=2.5k 차원 기준 re-permute **≤ 5분**(오프라인), 오버헤드 << 누적 절감.
* **재현성**: Docker/conda 환경 잠금, seed/클럭 고정 SOP.
* **관측성**: 이벤트/지표 스키마, 오버헤드 < 1%.
* **계측 자동화**: 벤치마크/클러스터링 결과가 `metrics.json` 및 `correlation/` 아티팩트로 저장되고, PRD 게이트(Lat −20%, L2 +10%p, EpT −15%) 자동 평가.
* **안전성**: 실패 시 자동 롤백(기존 레이아웃), 로그에 원인 기록.

## 7) 아키텍처 개요(상위 블록)

* **core/**: 그래프 생성(접근 공빈도 → W), 비용모델(C(π), Q), solver(스펙트럴+로컬).
* **adapters/**: S/Q/K 변환 어댑터(HDS 2:4, PTQ/FP8, KV 압축).
* **runtime/**: 실행 파이프라인, permutation 캐시, 실패 복구.
* **measure/**: Nsight/NVML 래퍼, 리포터(HTML/CSV).
* **bridge/**: TVM/StableHLO Pass PoC(선택).

## 8) 마일스톤/일정(기본 3주)

* **W1**: core/mock/measure 스캐폴드, 비용모델·solver 유닛테스트, CLI 초안.
* **W2**: S/Q/K 루프 통합, L2/Latency/EpT 측정 E2E, 리포트 자동화.
* **W3**: 모델 스냅샷 검증(SSM/Transformer), 문서/재현 패키지, 성능 목표 검수.

## 9) 의존성/전제

* **환경**: Python 3.10, (mps|cuda|cpu) 가드, CUDA 11.x+, Nsight Compute, NVML.
* **자원**: A100/H100 1대 이상(프로파일 권장), 퍼블릭 데이터셋(PTB/WikiText/SST-2).
* **리스크 공유**: 드라이버/클럭 변동 → SOP로 제어.

## 10) 리스크/완화책

* **R1 측정 변동성**: 전력/캐시 지표 노이즈 → 고정클럭·워밍업·N회 반복, CI 마이크로벤치 게이트.
* **R2 스케일**: 스펙트럴 O(D³) → 부분 고유값/샘플링/블록 병렬, 상한시간 설정.
* **R3 일반화**: 2개 워크로드 편향 → Ablation 스윕(희소율/정밀도/시퀀스 길이), 추가 모델 1종 옵션.
* **R4 통합비용**: 프레임워크 종속 → API/CLI 비침투, IR-bridge는 PoC로 한정.

## 11) 비기능적 품질/거버넌스

* **문서**: SAS/ICD/Pass-Doc/Cost-Spec/SOP/Repro-Pack 생성.
* **테스트**: 유닛/통합/E2E/성능/회귀. 결정론·숫자안정성 케이스 포함.
* **보안/라이선스**: SBOM, OSS 라이선스 준수, 데이터 사용정책.

## 12) 결정 매트릭스(접근 옵션)

| 옵션                                           | 효과                   | 리스크        | 비용   | 복잡도 | 선택     |
| -------------------------------------------- | -------------------- | ---------- | ---- | --- | ------ |
| **A** Iterative re-permute(S/Q/K) + 측정 파이프라인 | L2/Latency/EpT 직접 개선 | solver 스케일 | 중    | 중   | **採用** |
| B 비용모델만 추가(순차 파이프라인 유지)                      | 구현 용이                | 개선 한계      | 저    | 저   | 보류     |
| C 컴파일러 통합 선행(IR/Pass)                        | 장기 확장                | 초기 속도 저하   | 중\~상 | 상   | 후속     |

## 13) 실행 체크리스트

* [ ] PRD 승인(Owner/Reviewer 서명)
* [ ] SOP/테스트 플랜 확정(고정클럭/반복/워밍업)
* [ ] Latency/L2/EpT 자동 리포트 스크립트 완료
* [ ] 대표 워크로드 2종 결과가 목표 충족
* [ ] Repro 패키지(원시 로그 포함) 외부 검증 통과

## 14) 역할/책임(RACI)

* **Owner/PM**: 목표·범위·지표 승인, 리스크 관리
* **Tech Lead**: 아키/코어 설계, 성능 책임
* **Systems Eng.**: 런타임/측정, 관측성
* **ML Eng.**: S/Q/K 모듈, 데이터/모델 카드
* **Compiler Eng.**(옵션): IR-bridge PoC
* **QE**: SOP/재현/회귀 테스트

---

### 부록 A — 용어(간단)

* **S/Q/K**: Sparsity / Quantization / KV-cache compression
* **Q(모듈러리티)**: 커뮤니티 내 밀집도 측정 지표
* **EpT**: Energy per Token

---

**완료 기준 요약**
대표 2 워크로드에서 **Latency −20%, L2 +10%p, EpT −15%** 달성, 리포트/재현물/문서 세트 완료 → **MVP 출하**.
