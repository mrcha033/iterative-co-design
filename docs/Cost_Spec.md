# Cost Spec — Memory-Aware Layout Re-Optimization

**한 줄 요약**
공접근 가중치 $W$와 순열 $\pi$를 받아 **캐시 거리 비용 $C(\pi)$**, **모듈러리티 $Q(\pi)$**, **정렬/얼라인 정칙화 $R(\pi)$** 를 결합한 목적함수 $J(\pi)$를 정의하고, 하드웨어에 **교정(calibration)** 해석을 통해 **L2 hit·Latency·EpT** 개선과 직접 연결한다.

---

## 1) 가정·제약

* **범위**: 추론 경로. 텐서 feature 차원(또는 블록) 내 1D 순열을 주 대상으로 하되, multi-axis는 블록/태그로 축약.
* **입력**: $W\in\mathbb{R}_{\ge 0}^{D\times D}$ (대칭·희소), 초기/직전 순열 $\pi_0$.
* **계측**: Nsight Compute L2 hit 등 표준 메트릭 사용(섹터 기반). ([NVIDIA Docs][1])
* **일반성**: $Q$는 가중 네트워크용 모듈러리티 정의 채택(**\*\*2006\*\*** 고전 정의). 해상도 한계를 피하려면 $\gamma$ 도입(멀티해상도). ([PMC][2], [PubMed][3], [PNAS][4], [Wikipedia][5])

---

## 2) 목적함수(Top-level)

$$
\min_{\pi\in \mathfrak{S}_D}~ J(\pi)\;=\;\alpha~\underbrace{\widetilde{C}(\pi)}_{\text{캐시 거리}}\;+\;\beta~\underbrace{R_{\text{align}}(\pi)}_{\text{얼라인/벡터화}}\;+\;\gamma~\underbrace{R_{\text{stability}}(\pi;\pi_0)}_{\text{안정성}}\;-\;\mu~\underbrace{\widetilde{Q}(\pi)}_{\text{모듈러리티}}
$$

* $\alpha,\beta,\gamma,\mu$는 HW 교정으로 정함(§6).
* 틀: **거리↓ + 정렬↑ + 안정성↑ + 커뮤니티 응집↑** ⇒ **메모리 지역성↑ → L2 hit↑ → Latency/EpT↓** (I/O-aware 원리). ([arXiv][6], [NeurIPS Proceedings][7])

---

## 3) 구성요소 정의

### 3.1 캐시-거리 비용 $C(\pi)$

* 위치 함수: $pos_\pi(i)\in\{0,\dots,D-1\}$.
* **그룹화 단위** $g$: 한 캐시라인/워프-코얼레스 단위의 원소 수(예: line bytes / element size).
* **거리 커널** $\phi(d)$ (허버/포화형):

  $$
  d_{ij}=\Big\lfloor\frac{|pos_\pi(i)-pos_\pi(j)|}{g}\Big\rfloor,\quad 
  \phi(d)=\begin{cases}
  d, & d\le \lambda\\
  \lambda + \tau(d-\lambda), & d>\lambda,\ \ 0<\tau<1
  \end{cases}
  $$

  $$
  C(\pi)=\sum_{i<j} W_{ij}\,\phi(d_{ij})
  $$

  — **의미**: 같은 라인/타일 안에 묶일수록 비용↓. 장거리 벌점은 포화(멀리 가면 모두 비슷하게 나쁨).
* **정규화**: $\widetilde{C}(\pi)=C(\pi)/C_{\max}$, $C_{\max}=\phi(d_{\max})\sum_{i<j}W_{ij}$, $d_{\max}=\lfloor (D-1)/g\rfloor$.

> L2는 32B-섹터/라인 단위로 집계되며 **히트율↑ ⇒ 후속 메모리 단계 접근↓**. $\phi$는 이 섹터/라인 현실을 근사하는 가중. ([NVIDIA Docs][1])

### 3.2 모듈러리티 $Q(\pi)$ (가중 네트워크, 해상도 $\gamma$)

* 차수 $k_i=\sum_j W_{ij}$, $2m=\sum_{i}k_i$.
* **블록화 규칙**: $\pi$를 기준으로 연속 구간을 $k$ 블록 $B_1,\dots,B_k$로 슬라이스(고정 길이 또는 휴리스틱).
* **정의**:

  $$
  Q_\gamma(\pi)=\frac{1}{2m}\sum_{i,j}\left(W_{ij}-\gamma\frac{k_i k_j}{2m}\right)\mathbf{1}\{i,j\in B_c\ \text{for some }c\}
  $$

  — $\gamma=1$은 표준, $\gamma>1$은 **작은 커뮤니티 민감도↑**(해상도 한계 완화).
* **정규화**: $\widetilde{Q}=(Q_\gamma-\mathbb{E}_{\text{null}}[Q_\gamma])/(Q_{\max}-\mathbb{E}_{\text{null}}[Q_\gamma])$ 근사. ([PMC][2], [Wikipedia][5])

### 3.3 얼라인/벡터화 정칙화 $R_{\text{align}}$

* **벡터 폭** $v\in\{8,16,32\}$ 원소 단위, **얼라인** $a$ 바이트.
* 간단 근사:

  $$
  R_{\text{align}}(\pi)=\frac{1}{\sum_{i<j} W_{ij}}\sum_{i<j}W_{ij}\cdot \mathbf{1}\{pos_\pi(i)\bmod v \ne pos_\pi(j)\bmod v\}
  $$

  — 같은 타일 내부는 같은 모듈로 계급이면 코얼레스 ↑.

### 3.4 안정성 정칙화 $R_{\text{stability}}$

* **변동 억제**(재배치 비용/캐시 재생성 비용 반영):

  $$
  R_{\text{stability}}(\pi;\pi_0)=\frac{1}{D}\sum_{i} \mathbf{1}\{|pos_\pi(i)-pos_{\pi_0}(i)|>h\}
  $$

  — 히스테리시스 임계 $h$ 제공.

---

## 4) 제약·수렴·수치 가드

* **제약**: shape/dtype/stride 합법성(패스 T3에서 보장).
* **정지조건**: $\Delta J < \varepsilon$ 또는 시간상한(기본 300s). 향상 실패 시 $\pi\leftarrow\pi_0$ 롤백.
* **결정론**: 동일 seed/클럭 → 동일 $\pi$ (±ε).
* **복잡도**: $C$는 $O(\text{nnz}(W))$. $Q$는 블록 경계 누적 $O(\text{nnz}+kD)$.

---

## 5) 튜닝 파라미터(디폴트·가이드)

* $g$ (라인/타일 원소수): **HW에서 도출**(예: 128B 라인 & fp16 → $g=64$).
* $\lambda, \tau$: $\lambda$는 **타일 경계** 근처(예: 2–4), $\tau\in[0.1,0.3]$.
* $\gamma$: 1(기본)→1.5(작은 커뮤니티 보강).
* 가중치: $\alpha=1$ 고정, $\mu\sim 0.5$부터 시작, $\beta\in[0.1,0.3]$, $\gamma\in[0.0,0.2]$ (교정 후 확정).

---

## 6) 교정(Calibration): 지표↔목적함수 매핑

**목표**: $\Delta \widetilde{C}, \Delta \widetilde{Q}, R$ 변화가 **L2 hit/Latency/EpT** 변화를 얼마나 설명하는지 추정.

1. **파일럿 러너** $N\approx 20$개(랜덤/스펙트럴/로컬 탐색 조합) 실행.
2. **회귀**(RIDGE):

   $$
   \Delta \text{L2} = a_0 + a_1(-\Delta \widetilde{C}) + a_2(\Delta \widetilde{Q}) + a_3(-\Delta R_{\text{align}}) + \varepsilon
   $$

   $$
   \Delta \text{Latency} = b_0 + b_1(-\Delta \widetilde{C}) + b_2(\Delta \text{L2}) + \varepsilon
   $$

   $$
   \Delta \text{EpT} = c_0 + c_1(\Delta \text{Latency}) + c_2(\Delta \text{L2}) + \varepsilon
   $$
3. **가중치 역추정**: $(\alpha,\beta,\mu)$를 회귀 계수에 비례하게 재설정(선형/베이지안 갱신).
4. **유효성**: 교차검증 $R^2$, 잔차 진단.
5. **결론 사용**: 파이프라인에서 $\Delta J$가 **$\Delta \text{L2}/\Delta \text{Latency}$** 기대 변화와 정합되는지 감시.

> L2 hit/섹터·스루풋 정의는 Nsight Compute 문서 기준(버전 별 동일 핵심 의미). ([NVIDIA Docs][1])
> I/O-aware 관점은 FlashAttention 계열이 정식화(메모리 계층 접근 최소화). ([arXiv][6], [NeurIPS Proceedings][7])
> 스케줄/비용모델 접근의 일반성은 Halide/TVM Ansor 문헌에 정립. ([ACM Digital Library][8], [DSpace@MIT][9], [USENIX][10], [ar5iv][11])

---

## 7) 구현 스펙(계약)

### 7.1 API 매핑 (ICD 준수)

* `fit_permutation(W, ...) -> (pi, stats)` 가 **$C, Q$** 계산을 내장하고, `stats={"C":..., "Q":..., "J":..., "improved":...}` 반환.
* `solver.cfg`에 $\alpha,\beta,\gamma,\mu,g,\lambda,\tau,\gamma$(modularity) 저장.

### 7.2 수치 안정성

* $W$ 정규화 옵션: `"none" | "row" | "sym"`. 기본 `sym`: $W\leftarrow D^{-1/2}WD^{-1/2}$.
* 빈도 0/NaN 방지: $\epsilon=10^{-9}$ 더하기.
* 희소 $W$: CSR, 상삼각만 저장/합산.

---

## 8) 단위 테스트(필수)

1. **단조성(블록 구조)**

   * 2-Block 인공 $W$: 블록 응집↑일수록 $\widetilde{C}\downarrow,\ \widetilde{Q}\uparrow$.
2. **스케일 불변**

   * $W\leftarrow aW$ (a>0)에서 $\arg\min J$ 불변, $\widetilde{C},\widetilde{Q}$ 값 불변 범주.
3. **라인/타일 민감도**

   * $g$ 증가 시 동일 $\pi$에서 $C$ 감소(타일 크기 커질수록 벌점 완화).
4. **결정론**

   * 같은 seed/time\_budget → 같은 $\pi$ 해시.
5. **교정 일관성**

   * 파일럿 회귀 $R^2_{\text{L2}}>0.6$ 가이드, 미만이면 파라미터 그리드 재탐색.
6. **IR 합법성 연계**

   * 레이아웃 변경 뒤 Pass T3가 stride/align 위반 0건(스냅샷 검사).

---

## 9) 성능 메모 / 리스크 / 대안

* **복잡도**: $C,Q$ 평가는 $O(\text{nnz}(W))$ 근사, 솔버 지배(스펙트럴+로컬).
* **리스크**: 모듈러리티 해상도 한계 → $\gamma$ 스윕·다중 블록 규칙. ([PNAS][4])
* **리스크**: 정렬 벌점이 과도하면 $\pi$ 다양성↓ → $\beta$ 상한, warm-restart.
* **대안**: Louvain/Leiden으로 블록 먼저 찾고(연속 배치), 그 위에서 1D 순열 미세조정(하이브리드). ([ResearchGate][12], [Wikipedia][13])

---

## 10) 파라미터 초기값(추천)

```yaml
cost:
  alpha: 1.0
  beta: 0.2
  gamma_stability: 0.1
  mu: 0.5
  g: 64          # line_elems (예: 128B line, fp16)
  lambda: 3
  tau: 0.25
  modularity_gamma: 1.2
  blocks_k: 4    # Q 계산용 연속 블록 수(휴리스틱)
```

---

## 11) 의사코드(평가 루틴)

```python
def eval_cost(W, pi, pi_prev, cfg):
    pos = invperm(pi)                         # O(D)
    g = cfg.g
    C = 0.0
    for (i, j, wij) in upper_triangle_nnz(W): # CSR iterate
        dij = abs(pos[i] - pos[j]) // g
        phi = dij if dij <= cfg.lambda else cfg.lambda + cfg.tau * (dij - cfg.lambda)
        C += wij * phi
    C_norm = C / (cfg.C_max or (phi_max(cfg, D, g) * sum_nnz(W)))

    Q = modularity_contiguous_blocks(W, pi, k=cfg.blocks_k, gamma=cfg.modularity_gamma)
    Q_norm = normalize_modularity(Q)          # null-model baseline approx.

    R_align = align_penalty(W, pos, v=cfg.vec_width)
    R_stab  = stability_penalty(pi, pi_prev, h=cfg.hysteresis)

    J = cfg.alpha * C_norm + cfg.beta * R_align + cfg.gamma_stability * R_stab - cfg.mu * Q_norm
    return J, {"C":C_norm, "Q":Q_norm, "R_align":R_align, "R_stab":R_stab, "J":J}
```

---

## 12) 수용 기준(이 스펙 관점)

* 단위·통합 테스트(§8) 100% 통과.
* 파일럿 교정 후, **$\Delta \widetilde{C}\downarrow,\ \Delta \widetilde{Q}\uparrow$** 가 \*\*L2 hit↑/Latency↓/EpT↓\*\*와 **양의 상관**(p<0.05).
* Pass-pipeline 적용 시 **합법화 실패 0**, 결정론/시간상한 준수.

---

## 13) 출처(핵심 근거)

* **모듈러리티 정의/스펙트럴 접근**: Newman, **2006**, *PNAS*. ([PMC][2], [PubMed][3])
* **해상도 한계 & $\gamma$ 개념**: Fortunato & Barthélemy, **2007**, *PNAS*; RB/Leiden 문헌. ([PNAS][4], [Wikipedia][5])
* **I/O-aware 원리**(메모리 계층 접근 최소화): FlashAttention, 2022. ([arXiv][6], [NeurIPS Proceedings][7])
* **스케줄/비용모델 전통**: Halide(PLDI **2013**), TVM Ansor(OSDI 2020). ([ACM Digital Library][8], [DSpace@MIT][9], [USENIX][10], [ar5iv][11])
* **L2 측정 지표 정의**(섹터/히트율/스루풋): NVIDIA Nsight Compute 가이드. ([NVIDIA Docs][1])

---

[1]: https://docs.nvidia.com/nsight-compute/ProfilingGuide/?utm_source=chatgpt.com "2. Profiling Guide — NsightCompute 13.0 documentation"
[2]: https://pmc.ncbi.nlm.nih.gov/articles/PMC1482622/?utm_source=chatgpt.com "Modularity and community structure in networks - PMC"
[3]: https://pubmed.ncbi.nlm.nih.gov/16723398/?utm_source=chatgpt.com "Modularity and community structure in networks - PubMed"
[4]: https://www.pnas.org/doi/full/10.1073/pnas.0605965104?utm_source=chatgpt.com "Resolution limit in community detection | PNAS"
[5]: https://en.wikipedia.org/wiki/Leiden_algorithm?utm_source=chatgpt.com "Leiden algorithm"
[6]: https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com "[2205.14135] FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
[7]: https://proceedings.neurips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html?utm_source=chatgpt.com "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
[8]: https://dl.acm.org/doi/abs/10.1145/2499370.2462176?utm_source=chatgpt.com "Halide: a language and compiler for optimizing parallelism, locality, and recomputation in image processing pipelines: ACM SIGPLAN Notices: Vol 48, No 6"
[9]: https://dspace.mit.edu/handle/1721.1/85943?utm_source=chatgpt.com "Halide: a language and compiler for optimizing parallelism, locality, and recomputation in image processing pipelines"
[10]: https://www.usenix.org/conference/osdi20/presentation/zheng?utm_source=chatgpt.com "Ansor: Generating High-Performance Tensor Programs for Deep Learning | USENIX"
[11]: https://ar5iv.labs.arxiv.org/html/2006.06762v4?utm_source=chatgpt.com "[2006.06762] Ansor: Generating High-Performance Tensor Programs for Deep Learning"
[12]: https://www.researchgate.net/publication/1913681_Fast_Unfolding_of_Communities_in_Large_Networks?utm_source=chatgpt.com "(PDF) Fast Unfolding of Communities in Large Networks"
[13]: https://en.wikipedia.org/wiki/Louvain_method?utm_source=chatgpt.com "Louvain method"
