# Pass Design Doc — Iterative HW–SW Co-Design Layout Re-Optimization

**한 줄 요약**
StableHLO/TVM 파이프라인에 `permute → transform(S/Q/K) → re-permute`를 삽입하기 위한 **분석(Analysis)·변환(Transform) 패스**와 IR 메타데이터 계약·테스트·성능 가드를 규정한다.

---

## 0) 가정·제약

* **범위**: 추론(Inference) 경로. 학습/분산 스케줄링 제외.
* **IR**: StableHLO(MLIR 기반) 우선, TVM Unity는 동등 기능의 **Bridge**로 정의.
* **보장**: 수학적 동등성(semantic equivalence), 정밀도·스케줄 불변(변환에 의해 모델 품질 비열화 금지).
* **입력 신호**: `icd.transform_meta`(S/Q/K 변화), `icd.layout_perm`(π), `icd.coaccess_block`(블록 힌트).

---

## 1) 목표

1. **상태-변환 직후 레이아웃 재최적화**를 IR 수준에서 안전·결정론적으로 수행.
2. **캐시 지역성(=후속 커널의 메모리 접근 locality) 개선**을 유도하는 차원 재배치/스트라이드 조정.
3. IR 메타데이터를 커널/런타임까지 **왕복(round-trip)** 가능하게 유지.

---

## 2) Pass 파이프라인(DAG)

```
Frontend
  └─ attach-icd-metadata (A0)
      ├─ build-coaccess-graph (A1, Analysis)
      ├─ layout-cost-model (A2, Analysis)
      ├─ decide-permutation (T1, Transform - pre)
      ├─ apply-transform S|Q|K (EXT, 외부)
      ├─ rebuild-coaccess-graph (A1′)
      ├─ redecide-permutation (T2, Transform - post)
      ├─ legalize-layout (T3, Transform)
      └─ verify-and-annotate (V1, Verification)
Lowering → Codegen → Runtime
```

* **A0**: 메타태그 삽입(없으면 no-op).
* **A1/A1′**: 공접근 그래프 $W$ 구성(추적/옵션 기반).
* **A2**: 비용모델/모듈러리티 계산 준비.
* **T1/T2**: (재)퍼뮤테이션 결정.
* **T3**: 형상/스트라이드 합법화(reshape/transposed ops 조정).
* **V1**: 동치성·정렬·얼라인·앨리어싱(별도 분석) 검증.

---

## 3) Pass별 사양

### A0. `attach-icd-metadata`

* **입력**: 모듈/함수, 사용자 설정/ICD.
* **출력**: 텐서에 속성 부여

  * `icd.layout_perm : i32[D]`
  * `icd.layout_tag : "icd/v1"`
  * `icd.transform_meta : json` (S/Q/K)
  * `icd.coaccess_block : i32` (옵션)
* **합법성**: 타입/shape 불변, 속성만 추가.
* **실패**: 중복 태그 → 최신으로 교체(경고).

### A1. `build-coaccess-graph` (Analysis)

* **역할**: 연산자/텐서 **공접근 빈도** 기반 가중치 $W$ 산출.
* **방법**:

  * 기본: IR 내 **producer–consumer 경로**와 **reuse distance**/**fusion scope** 추정.
  * 옵션: 외부 추적 파일/샘플 실행 카운터 병합.
* **출력**: 모듈 속성 `icd.W`(CSR 포인터) 또는 side-car.
* **복잡도**: O(#edges + #uses).
* **실패**: 빈/희소 과도 → `W_min`(diagonal)로 폴백.

### A2. `layout-cost-model` (Analysis)

* **역할**: 비용 $C(\pi)=Σ_{i<j} W_{ij}|pos(i)-pos(j)|$ 및 **모듈러리티 Q** 계산 함수 제공.
* **출력**: 함수 핸들/테이블(옵션 캐시).
* **검증**: 블록 구조 강할수록 $C↓, Q↑$ 단조 테스트.

### T1. `decide-permutation` (Transform – pre)

* **트리거**: 초기 permute 또는 `icd.layout_perm` 부재.
* **알고리즘**: Fiedler ordering → local refine(2-opt/adj-swap, time\_budget).
* **출력**: `icd.layout_perm=π₀`.
* **보장**: shape 불변, stride/메모리오더 태그만 변경.
* **실패**: 시간 초과 → 초기해(정렬 전) 또는 기존 π 방치(플래그).

### EXT. `apply S|Q|K` (외부 변환)

* **역할**: sparsity/quant/KV 압축 수행(외부 단계).
* **계약**: 완료 후 `icd.transform_meta.delta_layout=true` 면 **T2 호출**.

### T2. `redecide-permutation` (Transform – post)

* **트리거**: `icd.transform_meta.delta_layout=true`.
* **역할**: 상태 변화 반영한 $W′$로 재퍼뮤테이션 → `π₁`.
* **정지조건**: `C(π₁) < C(π₀) * (1-ε)` 실패 시 **롤백**(π₀ 유지).
* **산출**: 개선 통계(ΔC, ΔQ)와 이벤트 기록.

### T3. `legalize-layout`

* **역할**: 레이아웃 변경으로 인한 **합법화**(canonicalize).

  * `transpose/reshape` 체인 축약, 브로드캐스트/패드 정합,
  * vectorization 친화 정렬(align=16/32/64) 메타 적용,
  * stride 충돌/alias 금지.
* **검증**: shape·dtype 불변, 정밀도 보존, 정렬 위반 0.
* **실패**: 불가능한 합법화 → 이전 레이아웃 롤백.

### V1. `verify-and-annotate`

* **역할**: 패스 후 불변식 검증·메트릭 주석화.

  * MLIR verifier + 커스텀: alias, layout-tag 일관성, 정렬/패딩, `π` 유효성(치환).
  * `icd.metrics.{Q,C,π_hash}` 부여.

---

## 4) IR 예시 (요약)

### 4.1 StableHLO (개념적 예)

**Before**

```mlir
%k = "stablehlo.dot"(%q, %v) : (tensor<[B,S,H,D]xf16>, tensor<[S,H,D]xf16>) -> tensor<[B,S,H]xf16>
%o = "stablehlo.add"(%k, %bias) : ...
```

**After (메타만 변경, 필요 시 transpose 삽입)**

```mlir
%q {icd.layout_perm = dense<[0,2,1,3]> : tensor<4xi32>} // π: (B,H,S,D)
%v {icd.layout_perm = dense<[1,2,0]>  : tensor<3xi32>}  // π: (H,D,S)
%k = "stablehlo.dot"(%q, %v) : ...
"icd.layout_annot"() {Q=0.47, C=9.0e6} : () -> ()
```

### 4.2 TVM Unity (스케치)

**Before**

```python
with Dataflow():
  q = relax.call_tir("matmul", (A,B), out_sinfo=...)
  o = relax.call_tir("add", (q,bias), ...)
```

**After**

```python
q = annotate_layout(q, perm=[0,2,1,3], tag="icd/v1")
v = annotate_layout(v, perm=[1,2,0], tag="icd/v1")
q, v = legalize_layout(q, v)  # 필요 시 transpose 삽입 후 fuse
```

---

## 5) 안전·합법성(Invariants)

* **형상/연산 동등성**: outputs 동일(수치 오차 ≤ 1e-6 상대).
* **정밀도**: dtype·quant 스케일 불변(quant meta 보존).
* **얼라인/스트라이드**: 벡터화 유효 얼라인 보장(≥16byte).
* **앨리어싱 금지**: view/take/transpose 체인 추적해 alias 충돌 금지.
* **결정론**: 동일 `seed/time_budget` → 동일 π(±ε).

---

## 6) 구성(Flags)

```yaml
passes:
  build_coaccess:
    source: "trace|static"
    fuse_scope: "block|function"
  decide_permutation:
    time_budget_s: 300
    refine_steps: 2000
    epsilon: 0.05
  legalize_layout:
    align: 32
    fuse_transpose: true
  verify:
    strict: true
```

---

## 7) 복잡도·성능

* **A1/A1′**: O(#uses) \~ 선형.
* **T1/T2**: 스펙트럴(부분 고유값) + 로컬 탐색 → 시간 상한 **하드 캡**.
* **T3**: 그래프 재작성 선형\~준선형.
* **오버헤드 가드**: `time_budget_s` 기본 300s, `improved` false면 **자동 롤백**.

---

## 8) 실패/폴백 시나리오

* `W` 희소 과도/수치 불안 → 균등 π 사용(경고), pass 종료.
* 합법화 실패 → 이전 π 롤백, 원인 기록.
* transform meta 불일치 → re-permute 스킵.
* ncu/NVML 미가용(외부 측정) → pass 진행, 측정만 비활성.

---

## 9) 테스트 계획(필수)

### 9.1 FileCheck 스타일(StableHLO)

* **Attach/Propagate**: `icd.layout_perm`가 생산자→소비자에 일관 주석되는지.
* **Legalize**: `transpose+reshape` 체인 축약 확인.

### 9.2 골든 IR 스냅샷

* **Dense vs Iterative** 전/후 동일 출력, `π` 치환성 검사.

### 9.3 수치 동등성

* 난수 입력 100 케이스, 상대오차 ≤ 1e-6.

### 9.4 성능 회귀

* 마이크로벤치(매트멈/어텐션)에서 **전/후 L2-proxy** 감소(≥10%) 또는 `Q↑`.

### 9.5 결정론

* 동일 seed/시간상한 → 동일 π 해시.

---

## 10) 관측성(메트릭 주입)

* Pass 종료 시 `module` 속성:

  * `icd.metrics = { "Q0":f, "Q1":f, "C0":f, "C1":f, "pi_hash":"…" }`
* 이벤트 로그: `stage, elapsed_ms, improved, reason`.

---

## 11) 상호작용(다른 패스와의 의존)

* **앞단**: shape/dtype infer → attach-icd.
* **뒤단**: fusion/vectorize 전에 **legalize-layout** 완료.
* **충돌 회피**: CSE/Canon 패스가 layout 주석 제거하지 않도록 보존 규칙 추가.

---

## 12) 보안/컴플라이언스

* 추적 기반 $W$ 생성 시 **개인정보/실데이터** 제거.
* 메타데이터에 민감값 금지(경로/계정 등).

---

## 13) 수용 기준(Acceptance)

* 두 워크로드(SSM/Transformer)에서 **T2 적용 후**

  * `Q1 > Q0`, `C1 < C0*(1-ε)`
  * 수치 동등성 패스, 합법화 실패 0건, 결정론 테스트 통과.
* IR 파이프라인을 켠/껐을 때 **컴파일 타임** ≤ +10% 증가(기본 설정).

---

## 14) 구현 노트(요지)

* 스펙트럴: Lanczos/LOBPCG(부분 고유값) 사용, 대규모는 샘플링.
* 로컬탐색: 인접 스왑 우선, 2-opt는 시간상한 내 제한.
* TVM Bridge: layout tag → `transform_layout`/`schedule.bind` 힌트로 투영.

---

## 15) 오픈 이슈(로드맵)

* **KV-cache 특화 $W$** (시퀀스 길이/배치 변화 반영).
* **PIM/CIM 제약 반영**(미래 확장): layout-cost에 열/대역 제약 가중.
* **학습형 비용모델 플러그인**(BO/RL) 선택 모드.

---

## 16) 참고(간단 의사코드)

```python
def repermute_after_transform(mod):
    W  = build_coaccess(mod)          # A1′
    Q0, C0, pi0 = metrics_from(mod)
    pi1 = spectral_order(W)
    pi1 = local_refine(W, pi1, time_budget_s=cfg.time_budget)
    C1, Q1 = cost(W, pi1), modularity(W, pi1)

    if C1 < C0 * (1 - cfg.epsilon):
        mod = apply_layout(mod, pi1)  # T2
        mod = legalize_layout(mod)    # T3
        annotate(mod, Q0,Q1,C0,C1,pi1)
    else:
        annotate_no_improve(mod, Q0,Q1,C0,C1)
    verify(mod)                        # V1
    return mod
```

---

## 17) 체크리스트(릴리즈 전)

* [ ] FileCheck 케이스 20개 통과
* [ ] 수치 동등성 100케이스, 실패 0
* [ ] 결정론/시간상한 테스트 통과
* [ ] 성능 회귀 게이트(Q↑/C↓) 통과
* [ ] IR 메타 왕복 테스트(attach→lower) 통과

---

필요 시 **SOP(측정 표준작업서)**, **Cost Spec(비용모델/튜닝 규정)**, **테스트 플랜**을 바로 잇는다.
