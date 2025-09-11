# ICD — Interface Control Document

*Iterative HW–SW Co-Design: Memory-Aware Layout Re-Optimization*

---

## 1) 한 줄 요약

이 문서는 `icd` CLI와 Python 라이브러리의 **공식 인터페이스 계약**을 규정한다. 입력/출력 스키마, 예외·에러코드, 버전·호환성, 결정론/성능 가드, 로깅·아티팩트 형식을 명세한다. (대상: ML/시스템/컴파일러/IR 엔지니어, 테스트/릴리즈 PM)

---

## 2) 가정·제약

* 언어/런타임: **Python 3.10**. 선택적 PyTorch(장치 가드), TVM/StableHLO는 PoC 플러그인.
* 하드웨어: 단일 GPU(A100/H100 권장). Nsight Compute(ncu), NVML 전력 읽기 선택.
* 범위: **추론** 경로(학습·분산 스케줄 제외).
* 버전: semver. 초기 릴리즈 `v0.1.0`(실험적).

---

## 3) 산출물(인터페이스 규격)

참고: 실행 방법과 산출물 예시는 docs/USAGE.md 를 참고하세요.

### 3.1 용어·기본 타입

* `D`: 상태/특징 차원(int).
* `W`: 공접근 가중치(대칭/비음수/희소 CSR).
* `π`(pi): permutation, `int32[D]`(0..D-1의 치환).
* `Q`: 모듈러리티 스칼라(float64).
* `C`: 캐시-프록시 비용 스칼라(float64).
* `RunId`: 실행 해시(내용 기반), `sha256[:12]`.
* 시간 단위: ms(지연), s(샘플링), Hz(주파수), W(전력), J/token(EpT).

---

### 3.2 Python API (안정 계약)

모든 API는 **동기**. 예외는 아래 “3.7 오류·예외” 준수.

```python
# graph
def build_w(
    source: Literal["trace", "mock"],
    *,
    trace: Optional[Iterable[tuple[int, int, float]]] = None,  # (i,j,weight)
    D: Optional[int] = None,
    blocks: int = 4,
    noise: float = 0.02,
    seed: int = 0,
    normalize: Literal["none", "row", "sym"] = "sym",
) -> "csr_matrix":
    """입력 트레이스 또는 모의 파라미터로 공접근 가중치 행렬 W 생성.
    계약: 결정론(동일 입력→동일 W), 대각 0, 음수 금지, CSR 반환."""

def fit_permutation(
    W: "csr_matrix",
    *,
    time_budget_s: int = 300,
    refine_steps: int = 2_000,
    k_blocks: Optional[int] = None,      # 모듈러리티 집단 수 힌트
    rng_seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """스펙트럴(Fiedler 정렬) 초기화 + 로컬 탐색.
    반환: (π:int32[D], stats: {"Q":float, "C":float, "iters":int, "elapsed_s":float, "improved":bool})"""

# S/Q/K adapters — 변환 후 메타데이터 반환(재퍼뮤트 트리거 판단에 사용)
def apply_sparsity(
    tensor_or_path, *,
    type: Literal["2:4", "unstructured"] = "2:4",
    rate: float = 0.5,
) -> tuple[Any, dict]:
    """meta = {"kind":"sparsity","type":"2:4","rate":0.5,"delta_layout":True}"""

def apply_quant(
    tensor_or_path, *,
    dtype: Literal["int8", "fp8"] = "int8",
    method: Literal["ptq-minmax", "ptq-kld"] = "ptq-minmax",
) -> tuple[Any, dict]:
    """meta = {"kind":"quant","dtype":"int8","method":"ptq-minmax","delta_layout":True}"""

def apply_kvcache(
    cache, *,
    block: int = 128,
    drop: float = 0.10,
) -> tuple[Any, dict]:
    """meta = {"kind":"kv","block":128,"drop":0.10,"delta_layout":True}"""

# orchestrator
@dataclass
class RunArtifacts:
    run_id: str
    out_dir: str
    metrics: dict   # 아래 3.4 스키마
    paths: dict     # 아티팩트 파일 경로

def run(config: dict) -> RunArtifacts:
    """파이프라인 실행:
    (1) W 생성 → (2) π0 → (3) Transform → (4) π1 → (5) 계측 → (6) 리포트/캐시.
    config 스키마는 3.3 참조."""

# 측정 유틸(직접 사용 가능)
def measure_latency(fn, *, repeats:int=1000, warmup:int=50, fixed_clock:bool=True) -> dict: ...
def measure_l2_hit(ncu_cmd: str, *, metrics: list[str]) -> dict: ...
def measure_power_nvml(*, sample_hz:int=10, duration_s:int) -> dict: ...
```

#### 결정론 규약

* `seed`·`rng_seed`·고정클럭 옵션이 같으면 **동일 π, 동일 W, ±ε 내 동일 결과**.
* `time_budget_s` 내 종료, 넘길 경우 `TimeoutError` (stats.improved=False, 초기해 반환 가능).

---

### 3.3 `run(config)` 입력 스키마 (YAML/JSON 동등)

```yaml
pipeline:
  mode: iterative              # linear | iterative
  repermute_on_delta: false    # if true, perform re-permute when transform_meta.delta_layout=true (even in linear)
  repeats: 1000
  warmup_iter: 50
  fixed_clock: true
graph:
  source: trace                # trace | mock
  trace: []                    # (i,j,weight) 리스트 또는 파일 경로
  mock: {D: 2560, blocks: 4, noise: 0.02, seed: 42}
  normalize: sym               # none|row|sym
solver:
  time_budget_s: 300
  refine_steps: 2000
  k_blocks: 4
  rng_seed: 0
transform:
  sparsity: {enable: true, type: "2:4", rate: 0.5}
  quant:    {enable: false, dtype: "int8", method: "ptq-minmax"}
  kv:       {enable: false, block: 128, drop: 0.10}
measure:
  ncu_enable: true
  ncu_metrics: ["l2_tex__t_sector_hit_rate.pct"]
  power_enable: true
  power_sample_hz: 10
report:
  out_dir: "runs/exp001"
  formats: ["html","csv"]
  # 지정 시 해당 포맷만 생성. 미지정 시 html/csv 모두 생성.
cache: {enable: false, cache_dir: ".icd_cache"}
```

* **계약**: `pipeline.mode=linear`인 경우 4단계(re-permute) 생략. 단, `repermute_on_delta=true`이고 변환 메타에 `delta_layout=true`인 경우에는 재퍼뮤트 1회 수행.
* **기본값**: 명시 없으면 PRD/SAS의 디폴트 사용.
* **캐시**: `cache.enable=true`이고 `cache.cache_dir`가 설정된 경우에만 활성화.
* **검증**: 스키마 위배 → `ConfigError`.

---

### 3.4 출력(메트릭/아티팩트) 스키마

**Metrics(JSON)**

```json
{
  "run_id": "a1b2c3d4e5f6",
  "hardware": {"gpu":"A100-40GB","driver":"535.xx"},
  "env": {"seed":0,"fixed_clock":true},
  "pipeline": {"mode":"iterative","repeats":1000,"warmup":50},
  "graph": {"D":2560,"nnz":123456,"normalize":"sym","source":"trace"},
  "solver": {"Q0":0.31,"C0":1.23e7,"Q1":0.47,"C1":9.01e6,"elapsed_s":212.4,"improved":true},
  "transform": {"sparsity":{"type":"2:4","rate":0.5},"quant":null,"kv":null},
  "latency_ms": {"mean": 12.8, "p50": 12.6, "p95": 13.5},
  "throughput_toks_s": 1843.2,
  "l2_hit_pct": 87.4,
  "ept_j_per_tok": 0.92,
  "quality_delta_pct": 0.03
}
```

**Artifacts(paths)**

```
{out}/
  W.csr.npz            # 공접근 가중치(JSON payload; no NumPy dep)
  w.meta.json          # 그래프 메타(공통; PyTorch는 ops.json 추가)
  perm_before.json     # π0
  perm_after.json      # π1 (iterative)
  ncu.json             # L2/프로파일 결과(옵션)
  power.csv            # 전력 샘플(옵션)
  report.html
  report.csv
  run.log              # 이벤트/오류
  config.lock.json     # 실제 사용된 확정 설정
```

**파일 포맷 계약**

* `perm*.json`: `{"D":2560,"pi":[...int...],"hash":"sha256"}`
* `W.csr.npz`: JSON payload with CSR arrays (`indptr, indices, data, shape, meta`), 대각 0, 비음수. (SciPy npz is an alternative for future native path.)
* `ncu.json`: ncu CLI 결과의 축약본(메트릭 이름→값 map).
* `report.*`: 전/후 비교 표/그림 포함(자동 생성).

---

### 3.5 CLI (안정 계약)

```
python -m icd.cli.main run -c config.json
  --override pipeline.mode=iterative
  --override solver.time_budget_s=180
  --out runs/exp001
```

옵션:

* `--dry-run`: 스키마 검증·계약 체크만 수행.
* `--print-schema`: 현재 버전의 입력 스키마 골격 덤프.
* `--no-measure`: 측정/리포트 단계를 건너뛰고 솔버만 실행.
* `--reuse-perm PATH`: 이전 π 재사용(perm_before.json 형식, 캐시/재사용 경로).
* (Planned) `--no-measure`: 솔버만 실행(벤치 제외).

**종료 코드**

* `0` 성공, `2` ConfigError, `3` ResourceError(GPU/도구 없음), `4` MeasureError, `5` SolverTimeout.

---

### 3.6 IR 브리지(StableHLO/TVM PoC) 메타데이터 계약

**Tensor/IR Attribute Keys**

* `icd.layout_perm`: `i32[D]` (π)
* `icd.layout_tag`: `"icd/v1"` (버전)
* `icd.coaccess_block`: `i32` (블록 힌트)
* `icd.transform_meta`: JSON 문자열(`{"sparsity":{"type":"2:4","rate":0.5},...}`)

**Pass 파이프라인 삽입 지점(예시)**

* Frontend → **attach-icd-metadata** → (opt) **icd-layout-pass** → Lowering.
* 계약: 메타태그가 없는 텐서는 **무시(no-op)**. 태그 불일치 시 경고 로그.

---

### 3.7 오류·예외(에러 택소노미)

| 코드          | 예외/상태                | 원인/설명                 | 처리            |
| ----------- | -------------------- | --------------------- | ------------- |
| `E_CFG_001` | `ConfigError`        | 스키마 위반/타입 불일치         | 즉시 중단         |
| `E_RES_101` | `ResourceError`      | GPU/드라이버/ncu/NVML 미가용 | 다운그레이드(가능)/중단 |
| `E_SLV_201` | `TimeoutError`       | `time_budget_s` 초과    | 초기해 반환 + 플래그  |
| `E_SLV_202` | `SolverNotConverged` | 수치 실패/NaN             | 초기해 롤백        |
| `E_ADP_301` | `TransformError`     | S/Q/K 적용 실패           | no-op 경고, 지속  |
| `E_MSR_401` | `MeasureError`       | ncu/NVML 실행 실패        | 벽시계 대체        |
| `E_IO_501`  | `ArtifactError`      | 파일 쓰기/권한              | 중단            |
| `E_INT_900` | `InternalError`      | 예상치 못한 예외             | 중단(버그 리포트)    |

**계약**: 실패 시 **롤백 가능 경로**가 있으면 유지하고, 리포트에 실패 원인·대체 경로 기록.

---

### 3.8 이벤트/로깅 스키마(Observability)

```json
{"ts":"2025-09-09T08:11:12Z","stage":"BUILD_W","ok":true,"meta":{"D":2560,"nnz":123456}}
{"ts":"2025-09-09T08:11:13Z","stage":"PERMUTE","ok":true,"meta":{"Q":0.31,"C":1.23e7,"elapsed_s":4.1}}
{"ts":"2025-09-09T08:11:30Z","stage":"TRANSFORM","ok":true,"meta":{"kind":"sparsity","rate":0.5}}
{"ts":"2025-09-09T08:15:02Z","stage":"REPERMUTE","ok":true,"meta":{"Q":0.47,"C":9.01e6}}
{"ts":"2025-09-09T08:15:20Z","stage":"MEASURE","ok":true,"meta":{"lat_ms":12.8,"l2_hit_pct":87.4,"ept":0.92}}
{"ts":"2025-09-09T08:15:21Z","stage":"REPORT","ok":true,"meta":{"out":"runs/exp001"}}
```

* 오버헤드 상한: **<1%**(샘플링 레벨 조정).
* 로그 레벨: `INFO/DEBUG/WARN/ERROR`. 민감정보 금지.

---

### 3.9 호환성/버전 정책

* **semver**: `MAJOR.MINOR.PATCH`.

  * MINOR에서 **새 키 추가** 가능(기본값 있어야 함).
  * MAJOR에서만 **키 제거/의미 변경** 가능.
* `config.lock.json`에 실제 키/기본값을 **동결 기록**.
* `icd.layout_tag`로 IR 메타 버전 고정.

---

### 3.10 캐시·재사용 계약

* 캐시 키: `hash(model_id, task_id, S/Q/K meta, D, seed, solver params)`
* 히트 시 `fit_permutation` 생략(검증 플래그로 재검증 가능).
* 호환 불일치 시 자동 무시(경고).

---

## 4) 테스트(계약 검증)

* **스키마 테스트**: 잘못된 타입/누락 필드 → `ConfigError` 이어야 한다.
* **결정론 테스트**: 동일 입력/seed/클럭 → π/메트릭 변동률 **≤ ε(1%)**.
* **에러 경로 테스트**: ncu 미가용 → `MeasureError` 후 벽시계로 대체.
* **성능 계약 테스트**: D=2.5k, `time_budget_s=300`에서 `elapsed_s ≤ 300`, `stats.improved` True 빈도 ≥ 95%.
* **IR 태그 왕복 테스트**: attach→pass→lower에서 `icd.layout_perm` 보존.

---

## 5) 성능 메모/리스크/대안

* **스펙트럴 비용**: O(D³) → 부분 고유값/샘플링/블록 분할로 완화.
* **계측 노이즈**: 고정클럭·워밍업·반복 N(≥1000) 강제.
* **IR 다양성**: 초기엔 **메타 태그 왕복**만, 본격 변환은 후속 Pass Doc에서 단계적 확장.

---

## 6) 부록 — JSON 스키마(발췌)

**입력 스키마(`run.config`)**

```json
{
  "type":"object",
  "properties":{
    "pipeline":{"type":"object","properties":{
      "mode":{"enum":["linear","iterative"]},
      "repeats":{"type":"integer","minimum":1},
      "warmup_iter":{"type":"integer","minimum":0},
      "fixed_clock":{"type":"boolean"}
    },"required":["mode"]},
    "graph":{"type":"object","properties":{
      "source":{"enum":["trace","mock"]},
      "trace":{"anyOf":[
        {"type":"array","items":{"type":"array","items":[{"type":"integer"},{"type":"integer"},{"type":"number"}], "minItems":3,"maxItems":3}},
        {"type":"string"}]},
      "mock":{"type":"object","properties":{
        "D":{"type":"integer","minimum":2},
        "blocks":{"type":"integer","minimum":1},
        "noise":{"type":"number","minimum":0.0},
        "seed":{"type":"integer"}
      }},
      "normalize":{"enum":["none","row","sym"]}
    },"required":["source"]},
    "solver":{"type":"object","properties":{
      "time_budget_s":{"type":"integer","minimum":1},
      "refine_steps":{"type":"integer","minimum":0},
      "k_blocks":{"type":["integer","null"]},
      "rng_seed":{"type":"integer"}
    }},
    "transform":{"type":"object"},
    "measure":{"type":"object"},
    "report":{"type":"object","properties":{
      "out_dir":{"type":"string"},
      "formats":{"type":"array","items":{"enum":["html","csv"]}}
    }}
  },
  "required":["pipeline","graph","solver","report"]
}
```

**출력 메트릭 스키마**는 3.4 참조.

---

### 7) 관련 문서 링크

* **PRD**: 목표/지표/수용 기준
* **SAS**: 컴포넌트·데이터·플로·예산
* **SOP(측정 표준작업서)**: 고정클럭·워밍업·반복 규칙
* **Pass Design Doc**: IR 삽입 지점·변환 규칙
* **Cost Spec**: C(π)·Q 정의와 튜닝 규칙
