# Repro Plan — Artifact Evaluation & 결과 재현 절차

**한 줄 요약**
동일 환경·입력·시드에서 `linear` vs `iterative` 전·후 성능을 **24h 내** 재현하도록, 설치→환경고정→데이터/모델→실험→검증→아티팩트 패키징을 자동화한다.

---

## 1) 가정·제약

* **목표**: PRD 목표(Latency −20%, L2 +10%p, EpT −15%)를 대표 2 워크로드(SSM/Mamba-3B, Transformer/BERT-base)에서 재현.
* **환경**: Python 3.10, CUDA 11.x+, A100/H100 1대 권장(ncu/NVML 가능). 로컬 Mock 재현 경로 포함(무GPU 가능).
* **데이터/모델**: 공개 체크포인트 사용(Hugging Face 등). 상용/비공개 데이터 없음.
* **결정론**: seed/클럭 고정, 워밍업 후 반복 측정. SOP 준수 전제.

---

## 2) 산출물(재현자에게 제공되는 것)

* **Repro 패키지**:

  * `Makefile`, `scripts/repro_{smoke,full,ae}.sh`, `scripts/collect_artifacts.py`
  * `env/environment.yml`(conda), `Dockerfile`(옵션)
  * `configs/{mock.yaml,mamba.yaml,bert.yaml}`
  * `docs/{SOP.md, TESTPLAN.md, PRD.md, SAS.md, ICD.md, PASS_DOC.md, COST_SPEC.md}`
* **기대 결과**: `runs/*/metrics.json`, `report.{html,csv}`, `ncu.json`(가능 시), `power.csv`, `config.lock.json`.

---

## 3) 사전 준비(재현자 액션)

### 3.1 소프트웨어 설치

```bash
git clone <repo> && cd <repo>
conda env create -f env/environment.yml && conda activate icd
# (옵션) Docker:
# docker build -t icd-repro -f Dockerfile .
```

### 3.2 하드웨어 점검

```bash
python - <<'PY'
import torch, subprocess, json
print("CUDA:", torch.cuda.is_available(), "Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("NVML:", subprocess.call(["nvidia-smi"])==0)
PY
```

* 가능하면 `nvidia-smi -pm 1` 로 퍼시스턴스 모드 on, 전력/클럭 고정은 권장(권한 없으면 “비고정”으로 기록).

### 3.3 모델/데이터 준비(자동 다운로드)

```bash
make fetch_models   # 내부 스크립트가 공개 체크포인트/데이터셋 캐시
```

* 네트워크 불가 시, 오프라인 캐시 디렉터리를 `ICD_DATA_DIR`로 지정.

---

## 4) 재현 시나리오(빠른→완전)

### 4.1 Repro 0 — Mock 스모크(10분 내, 무GPU 가능)

```bash
make repro_smoke
# 또는:
icd run -c configs/mock.yaml --override pipeline.mode=linear  --out runs/mock_linear
icd run -c configs/mock.yaml --override pipeline.mode=iterative --out runs/mock_iter
python scripts/collect_artifacts.py runs/mock_*
```

**판정**: `ΔC≤-10%` & `ΔQ≥0`(Cost Spec 기준). `report.html`에 상관 플롯 생성.

### 4.2 Repro 1 — Mamba-3B (SSM) 메인

```bash
icd run -c configs/mamba.yaml --override pipeline.mode=linear   --out runs/mamba_linear
icd run -c configs/mamba.yaml --override pipeline.mode=iterative --out runs/mamba_iter
```

**기대치**(스모크 게이트): Latency ≤ −10%, L2 ≥ +5%p, EpT ≤ −8%
**엄격 게이트**: PRD 기준(−20% / +10%p / −15%)

### 4.3 Repro 2 — BERT-base (Transformer)

```bash
icd run -c configs/bert.yaml --override pipeline.mode=linear    --out runs/bert_linear
icd run -c configs/bert.yaml --override pipeline.mode=iterative  --out runs/bert_iter
```

**기대치** 동일(스모크/엄격 게이트 분리).

### 4.4 Repro 3 — Ablation(선택, 2시간 내)

* 희소율 `{0.3, 0.5, 0.7}`, 정밀도 `{fp8,int8}`, 시퀀스 길이 `{256, 1024}` 스윕:

```bash
make repro_ablation   # 내부에서 매트릭스 조합 실행
```

### 4.5 Repro 4 — IR Pass PoC(선택)

* StableHLO/TVM 브리지 켠 모드로 동일 실험 1회:

```bash
icd run -c configs/mamba.yaml --override bridge.enable=true --out runs/mamba_bridge
```

**판정**: 수치 동등성(상대오차≤1e-6), 성능 변동 ±ε.

---

## 5) 검증·패키징

### 5.1 자동 검증

```bash
python scripts/validate_results.py runs/*_{linear,iter}
# 규칙: ΔLatency, ΔL2, ΔEpT CI 포함·판정; 결정론(π 해시·핵심 지표 변동 ≤1%)
```

### 5.2 아티팩트 패키징

```bash
python scripts/collect_artifacts.py runs/mamba_* runs/bert_* -o artifacts/icd_artifacts.zip
# 포함: metrics.json, ncu.json, power.csv, report.{html,csv}, config.lock.json, run.log
```

---

## 6) 구성 파일 예시(핵심 발췌)

```yaml
# configs/mamba.yaml
pipeline: {mode: iterative, repeats: 1000, warmup_iter: 50, fixed_clock: true}
graph: {source: trace, normalize: sym}
solver: {time_budget_s: 300, refine_steps: 2000, k_blocks: 4, rng_seed: 0}
transform: {sparsity: {enable: true, type: "2:4", rate: 0.5}, quant: {enable: false}}
measure: {ncu_enable: true, ncu_metrics: ["l2_tex__t_sector_hit_rate.pct"], power_enable: true, power_sample_hz: 10}
report: {out_dir: "runs/mamba_iter", formats: ["html","csv"]}
```

---

## 7) Makefile 타깃(발췌)

```makefile
.PHONY: env fetch_models repro_smoke repro_full repro_ae pack

env:
	conda env create -f env/environment.yml || true

fetch_models:
	python scripts/fetch_models.py --out $(HOME)/.cache/icd

repro_smoke:
	icd run -c configs/mock.yaml --override pipeline.mode=linear  --out runs/mock_linear
	icd run -c configs/mock.yaml --override pipeline.mode=iterative --out runs/mock_iter
	python scripts/validate_results.py runs/mock_*

repro_full: repro_mamba repro_bert
repro_mamba:
	icd run -c configs/mamba.yaml --override pipeline.mode=linear   --out runs/mamba_linear
	icd run -c configs/mamba.yaml --override pipeline.mode=iterative --out runs/mamba_iter
	python scripts/validate_results.py runs/mamba_*

repro_bert:
	icd run -c configs/bert.yaml --override pipeline.mode=linear    --out runs/bert_linear
	icd run -c configs/bert.yaml --override pipeline.mode=iterative  --out runs/bert_iter
	python scripts/validate_results.py runs/bert_*

pack:
	python scripts/collect_artifacts.py runs/* -o artifacts/icd_artifacts.zip
```

---

## 8) 기대 시간(참고, 장치·옵션에 따라 변동)

* **Mock 스모크**: \~10분
* **Mamba-3B / BERT**: 솔버 ≤ 5분/케이스(오프라인), 측정 루프(1000 반복) 수분대
* **Ablation**: 조합 수×(솔버+측정) 누적

---

## 9) 실패 시 대체 경로(Fallback)

* **ncu 미가용**: `--override measure.ncu_enable=false` → L2 미보고.
* **NVML 미가용**: `--override measure.power_enable=false` → EpT 미보고.
* **클럭 고정 불가**: `fixed_clock=false` 기록, 비교표에 “비고정” 뱃지, 결정론 게이트 완화.
* **Solver timeout**: 초기해 반환, `stats.improved=false` 플래그 표기, 한 번 재시도 후 중단.

---

## 10) 판정 기준(자동)

* **스모크 통과**:

  * Mock: `ΔC≤-10% & ΔQ≥0`
  * Mamba/BERT: Latency ≤ −10%, L2 ≥ +5%p, EpT ≤ −8%
* **최종 통과(엄격)**: PRD 기준(−20% / +10%p / −15%) **동시 만족**, 품질 저하 ≤ 0.1%p.

---

## 11) 로그/감사 추적

* 모든 실행에 대해 `run.log`(이벤트 스키마), `config.lock.json`(실제 사용 설정) 보존.
* `git rev-parse HEAD` 및 `conda list --explicit`를 `artifacts/manifest.txt`에 포함.

---

## 12) 보안/윤리

* 아티팩트에 **개인정보/경로/자격증명** 포함 금지.
* 모델/데이터 라이선스 표기, 재배포 금지 항목은 해시만 기록.

---

## 13) 트러블슈팅(요약)

* Latency 개선 없고 L2만 상승 → DRAM 대역/NoC 지표 확인(메모리 병목).
* 반복 편차↑ → 워밍업/배경부하/온도 확인, 반복수↑.
* π 변동 큼 → seed/시간상한/정렬 파라미터 재확인.
* 브리지 활성 시 성능↓ → `legalize-layout` 실패 여부·fuse 경로 변동 확인.

---

## 14) Repro 체크리스트(재현자용)

* [ ] `conda activate icd` / `Docker run` 완료
* [ ] `make fetch_models` 성공(오프라인이면 캐시 경로 설정)
* [ ] `repro_smoke` 통과(보고서 생성)
* [ ] `repro_full` 통과(스모크 게이트)
* [ ] PRD 게이트 충족(최종)
* [ ] `pack` 실행, `artifacts/icd_artifacts.zip` 제출

---
