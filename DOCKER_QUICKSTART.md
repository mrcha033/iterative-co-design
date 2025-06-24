# 🐳 Docker 빠른 시작 가이드

Iterative Co-Design 프로젝트를 Docker로 간편하게 실행하는 방법입니다.

## 📋 사전 요구사항

- Docker Desktop 또는 Docker Engine
- NVIDIA Docker (GPU 사용 시)
- 최소 8GB RAM, 10GB 디스크 공간

## 🚀 5분 빠른 시작

### 1단계: 저장소 클론
```bash
git clone https://github.com/yunmin-cha/iterative-co-design.git
cd iterative-co-design
```

### 2단계: Docker 이미지 빌드
```bash
docker-compose build base
```
⏱️ *첫 빌드는 5-10분 소요됩니다*

### 3단계: 환경 확인
```bash
docker-compose run base
```

**예상 출력:**
```
🔍 Environment Verification
===========================
PyTorch: 2.3.1+cu121
CUDA available: True
CUDA devices: 1
✅ BERT models: Available
✅ Mamba models: Available  # 또는 ❌ Not available

🚀 Ready to run experiments!
```

### 4단계: 실험 실행
```bash
# 안정적인 BERT 실험
docker-compose run trainer

# 고급 Mamba 실험 (GPU 필요)
docker-compose run mamba-trainer
```

## 📊 주요 명령어

| 목적 | 명령어 |
|------|--------|
| **환경 검증** | `docker-compose run base` |
| **BERT 실험** | `docker-compose run trainer` |
| **Mamba 실험** | `docker-compose run mamba-trainer` |
| **대화형 셸** | `docker-compose run shell` |
| **Jupyter 노트북** | `docker-compose up jupyter` |

## 🛠️ 커스텀 실험

원하는 실험 설정으로 실행하기:

```bash
# 예시: BERT + SST2 + iterative 방법
docker-compose run --rm shell bash -c "
  source /opt/conda/bin/activate iterative-co-design &&
  python scripts/run_experiment.py model=bert_base dataset=sst2 method=iterative
"

# 예시: Mamba + WikiText + dense 방법
docker-compose run --rm shell bash -c "
  source /opt/conda/bin/activate iterative-co-design &&
  python scripts/run_experiment.py model=mamba_3b dataset=wikitext103 method=dense
"
```

## 📁 결과 확인

실험 결과는 호스트의 다음 위치에 저장됩니다:
- `outputs/`: 실험 메트릭과 설정
- `results/`: 모델 체크포인트와 분석 결과

```bash
# 결과 확인
ls outputs/
ls results/
```

## 🔧 문제 해결

### GPU가 인식되지 않는 경우
```bash
# NVIDIA Docker 설치 확인
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi
```

### 메모리 부족 오류
```bash
# 사용하지 않는 이미지 정리
docker system prune -a
```

### Mamba 설치 실패
- 정상적인 현상입니다 (BERT 모델 사용 권장)
- A100 GPU 환경에서만 안정적으로 작동

## ⚡ 성능 팁

1. **SSD 사용**: Docker 빌드 속도 향상
2. **메모리 할당**: Docker Desktop에서 8GB+ 할당
3. **캐시 활용**: 이미지 재빌드 시 Docker 빌드 캐시 활용

## 📚 추가 정보

- 전체 문서: [README.md](README.md)
- 실험 설정: [configs/](configs/)
- 스크립트: [scripts/](scripts/)

---

🎯 **목표**: Docker를 통해 복잡한 환경 설정 없이 바로 실험 시작! 