# SOP — 측정 표준작업서 (Latency / L2 Hit / Energy-per-Token)

**한 줄 요약**
고정된 환경에서 `permute → transform(S/Q/K) → re-permute` 전·후를 동일 조건으로 재현 측정하고, Latency·L2 Hit·EpT(J/token)를 신뢰구간과 함께 산출·보관한다. 실패 시 즉시 롤백·재측정.

---

## 0) 가정·제약

* **범위**: 추론(inference) 측정. 학습/분산 제외.
* **HW/툴**: A100/H100 1대↑, Python 3.10, Nsight Compute(ncu CLI), NVML 전력 읽기 가능(또는 동등 보드 전력계).
* **연결 문서**: PRD(목표 지표) · SAS(블록/아키) · ICD(API/아티팩트) · Pass Doc(IR 경로) · Cost Spec(목적함수).

---

## 1) 산출물

* **메트릭 요약** `metrics.json`: Latency(ms){mean,p50,p95}, tokens/s, L2\_hit(%), EpT(J/token), 품질 변화(Δacc 등), 환경 해시.
* **원시 로그**: `ncu.json`(또는 section 보고서), `power.csv`(t, W), `run.log`(이벤트), `config.lock.json`.
* **리포트**: `report.html|csv` 전·후 비교 표/그림.

---

## 2) 환경 고정(필수 체크리스트)

* [ ] **GPU 상태 잠금**: persistence-mode on, 가능 시 **application clocks/전력캡 고정**.
* [ ] **드라이버/라이브러리 버전 스냅샷**: CUDA, cuDNN, 프레임워크 버전 기록.
* [ ] **온도 안정화/워밍업**: 워밍업 N\_warmup(기본 50) 수행.
* [ ] **RNG/Seed 고정**: 데이터 샘플·모델 초기화·샘플러 seed 통일.
* [ ] **배경 부하 차단**: 전용 노드/고정 CPU governor 권장.
* [ ] **입력 고정**: 동일 배치·시퀀스 길이·패딩 정책·KV 캐시 설정.

> 실패 시 **측정 무효**. 재측정 전에 환경부터 재고정.

---

## 3) 파일/경로 규약

```
{out}/
  config.lock.json   # 실제 사용 설정 스냅샷
  run.log            # 단계별 이벤트 로그
  ncu.json           # L2 등 프로파일 결과(가능 시)
  power.csv          # t(s), power_w
  metrics.json       # 요약 메트릭
  report.html|csv    # 전/후 비교
```

---

## 4) 단계별 절차

### 4.1 환경 지문 수집(Fingerprint)

1. GPU·드라이버·클럭·전력캡·ECC·온도·UUID를 쿼리 → `run.log`에 저장.
2. Git commit/브랜치, Docker/conda 환경, 패키지 버전, 모델/데이터셋 해시 기록.
3. **고정**이 안 되는 항목(클럭/전력 등)은 “비고정”으로 명시.

### 4.2 워밍업 & 안정화

1. 동일 입력으로 **워밍업 50회**(PRD 기본).
2. 온도/전력 흔들림이 수렴하는지 간단 체크(rolling std).

### 4.3 라틴시/스루풋 측정

1. `repeats=1000`으로 동일 입력 반복.
2. **벽시계 시간**(high-res monotonic)으로 per-iter latency 수집 → p50/p95 계산.
3. 총 토큰 처리량/시간으로 **tokens/s** 산출.
4. 결과는 `{lat_ms, toks_per_s}`로 `metrics.json`에 반영.

### 4.4 L2 Hit 측정

* **권장**: ncu **섹션 기반** 실행(예: Memory Workload/Cache 섹션)으로 L2 hit rate를 추출.
* **대체**: 특정 메트릭 이름을 직접 지정(아키텍처별 이름 상이 가능).
* 3회 이상 반복 실행해 평균값(±표준편차) 기록.
* ncu 오버헤드로 Latency가 크게 변하면, **L2 전용 러너**와 **Latency 전용 러너**를 분리 측정.

### 4.5 전력·EpT 측정

1. NVML(또는 보드 전력계)로 **고정 주기(기본 10Hz)** 샘플링.
2. 측정 창을 Latency 측정 루프와 동기화(시작/끝 타임스탬프).
3. 트라페zoidal 적분으로 **에너지\[J]** 계산 후 **EpT = 에너지 / 생성 토큰수**.
4. 3회 이상 반복, 평균·신뢰구간 보고.

### 4.6 전·후 비교 및 수용판정

1. **Baseline(Linear)** vs **Iterative** 를 **동일 조건**에서 각각 3회 이상 실행.
2. 각 지표의 Δ(차이)와 95% CI, 효과크기(g/Hedges) 기록.
3. PRD 기준 충족 여부 판단(예: Latency −20% / L2 +10%p / EpT −15%).
4. 미충족 시 원인분석(§6) 후 1회 재시도. 재시도 실패 시 **롤백**.

---

## 5) 명령 예시(참고 스니펫)

### 5.1 환경 잠금/기록(가능 시)

```bash
# 고정은 권한/모델에 따라 제약 있음. 실패해도 '비고정'으로 로깅.
nvidia-smi -pm 1
nvidia-smi --query-gpu=name,uuid,driver_version,pstate,clocks.gr,clocks.sm,clocks.mem,power.limit,temperature.gpu --format=csv -i 0
# (옵션) 전력캡/클럭 설정: 권한 필요
# nvidia-smi -pl 250
# nvidia-smi -ac <memMHz>,<smMHz>
```

### 5.2 파이프라인 실행(전·후)

```bash
# Baseline (linear)
icd run -c config.yaml --override pipeline.mode=linear --out runs/linear

# Iterative
icd run -c config.yaml --override pipeline.mode=iterative --out runs/iter
```

### 5.3 L2 측정(ncu)

```bash
# 섹션 기반(권장): Cache/Memory 섹션을 한 번에 수집
nv-nsight-cu-cli --section "MemoryWorkloadAnalysis" --section "MemoryChart" \
  --nvtx --kernel-name <your_kernel_regex> --page raw ./runner --config config.lock.json \
  --export json --export-file runs/iter/ncu.json
```

### 5.4 전력 샘플 로깅(파이썬 예시)

```python
# measure/power_logger.py
import time, csv
from measure.power_stub import read_power_w  # NVML or NaN
with open("runs/iter/power.csv","w",newline="") as f:
    w = csv.writer(f); w.writerow(["t_s","power_w"])
    t0 = time.perf_counter()
    for _ in range(int(10*60)):   # 10Hz * 60s 예시
        t = time.perf_counter()-t0
        w.writerow([f"{t:.6f}", f"{read_power_w():.3f}"])
        time.sleep(0.1)
```

---

## 6) 검증·품질관리(QA)

### 6.1 일관성·결정론

* 동일 입력/seed/클럭에서 **Latency 분산** CV ≤ 5% 권장.
* π/메트릭 해시가 세션 간 일치(±ε)하는지 확인.

### 6.2 신뢰구간·통계

* 반복 n≥3(권장 n≥5)로 평균·95% CI 산출.
* 성능 주장은 **Δ와 CI**를 같이 표기(“−22% \[−19, −25]”).

### 6.3 아웃라이어 처리

* IQR·MAD로 outlier 검출, 원인(스케줄링/jitter/열) 로그와 함께 **제외 여부 기록**.

### 6.4 오버헤드 가드

* ncu를 켠 상태의 Latency가 baseline 대비 **>10% 악화**되면 **분리 러너**로 전환.

### 6.5 실패/롤백 정책

* Latency/L2/EpT 중 **2개 이상 악화** → 자동 롤백(직전 π).
* 재측정 1회 후에도 악화 지속 → RCA 보고서 작성(§7).

---

## 7) RCA(원인 분석) 절차

1. **환경**: 드라이버/클럭/온도/전력 캡 변동?
2. **입력**: 배치/시퀀스/kvcache 세팅 변화?
3. **커널**: fused/unfused 차이? 다른 커널 경로로 바뀌었는가?
4. **메모리**: L2 hit는 올랐지만 Latency가 정체? → DRAM 대역/NoC 혼잡 지표 재확인.
5. **솔버**: 시간상한 타서 미개선? 초기해 품질? (Cost Spec의 ΔJ 확인)
6. **통계**: 반복 수 부족 or outlier 영향? → n 재측정.

---

## 8) 보안·윤리·컴플라이언스

* 프로파일/로그에 **데이터·자격증명·경로** 등 민감정보 금지.
* 데이터셋/모델/라이선스 준수 표기.
* 에너지 측정 보고 시 **측정 한계**(센서 샘플링, 계측 정확도) 명시.

---

## 9) 수용 기준(Acceptance)

* 대표 2 워크로드에서 **PRD 목표** 충족:

  * Latency −20% 이상, L2 +10%p 이상, EpT −15% 이상(품질 저하 ≤ 0.1%p).
* 아티팩트 완비: `metrics.json`, `ncu.json`(또는 동등), `power.csv`, `config.lock.json`, `report.*`
* 외부 검증자가 문서·스크립트만으로 **24h 내 재현** 성공.

---

## 10) 실패·대체 경로(Fallback)

* **NVML 불가**: 외부 전력계(보드/소켓) 사용 또는 EpT **미보고**(N/A) 명시.
* **ncu 불가**: L2 hit 미보고, Latency/Throughput만 보고(제한사항 명시).
* **클럭 고정 불가**: `fixed_clock=false`로 기록, CI에서 결과 **비교 불가**로 태깅.

---

## 11) 자동화 훅(ICD 연계)

* `icd run` 종료 시 자동으로:

  1. 환경 지문 저장 →
  2. 워밍업/측정 루프 →
  3. ncu/NVML 호출 →
  4. EpT 적분 →
  5. 전·후 비교 리포트 생성.

---

## 12) 부록 — 결과 표준 포맷(발췌)

**metrics.json (예)**

```json
{
  "run_id":"a1b2c3d4e5f6",
  "env":{"gpu":"A100-40GB","driver":"535.xx","fixed_clock":true,"seed":0},
  "pipeline":{"mode":"iterative","repeats":1000,"warmup":50},
  "latency_ms":{"mean":12.8,"p50":12.6,"p95":13.5},
  "throughput_toks_s":1843.2,
  "l2_hit_pct":87.4,
  "ept_j_per_tok":0.92,
  "quality_delta_pct":0.03
}
```

**power.csv (예)**

```
t_s,power_w
0.000000,210.5
0.100112,212.6
...
```

---

## 13) 실행 체크리스트(최종)

* [ ] 환경 잠금/지문 수집 완료
* [ ] 워밍업·반복 수 충족
* [ ] Latency/Throughput 기록
* [ ] L2 hit 수집(가능 시)
* [ ] EpT 적분 계산(가능 시)
* [ ] 전·후 비교 및 CI/효과크기 산출
* [ ] 아티팩트·리포트 저장 및 커밋

---

## 14) 리스크/대안

* **계측 노이즈**: 반복 수↑, 고정클럭, 프로파일 분리.
* **메트릭 명칭 상이**: 섹션 기반 수집으로 호환성 확보, 장치별 맵 테이블 유지.
* **오버헤드**: ncu 샘플링/범위 축소, 프로파일 전용 실행으로 분리.

---
