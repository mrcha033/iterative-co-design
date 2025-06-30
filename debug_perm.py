# debug_perm.py
import torch, copy
import sys
from pathlib import Path

# --- Add project src to python path ---
# This allows the script to find local modules like 'utils' and 'co_design'
project_root = Path(__file__).resolve().parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
# ---

from transformers import AutoModelForCausalLM, AutoTokenizer
from itertools import zip_longest
from collections import OrderedDict
from omegaconf import OmegaConf

from co_design.iasp import run_iasp_on_mamba   # <- 당신 프로젝트 import 경로
from utils.input import make_dummy_input      # <- 동일

MODEL_ID = "state-spaces/mamba-370m-hf"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1) 원본·사본 준비 -----------------------------------------------------------
orig = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
orig.eval()
perm = copy.deepcopy(orig)                    # 깊은 복사 후 permutation 적용
perm.eval()

# dummy dataloader 필요 없으면 간단히 한 배치만 사용
tok = AutoTokenizer.from_pretrained(MODEL_ID)
dummy_in = make_dummy_input(orig, tok, device)

# 2) IASP 실행  --------------------------------------------------------------
#   최소 샘플만 써서 빨리 돌림 (정확하진 않아도 permutation 자체는 만들어짐)
perm_tensor, _ = run_iasp_on_mamba(
    perm,
    dataloader=[dummy_in],                    # 리스트도 DataLoader 처럼 iterable
    iasp_config=OmegaConf.create({
        "max_samples": 512,
        "cluster_size_range": [16, 64],       # 빠른 실행용
        "knn_k": 64,
        "spectral_n_init": 10,
        "spectral_random_state": 42
    }),
)

# 3) hook 으로 층별 출력 비교 -----------------------------------------------
EPS = 1e-4
mismatch = OrderedDict()

def register_hooks(m1, m2, prefix=""):
    for (name1, module1), (name2, module2) in zip_longest(
        m1.named_children(), m2.named_children()
    ):
        # 서브 모듈이 없으면 leaf → 비교
        if module1 is None or module2 is None:
            continue
        full_name = f"{prefix}.{name1}" if prefix else name1

        def make_hook(tag):
            def _hook(mod, inp, out):
                # 버전별로 tuple 반환일 수도 있어 첫 원소만 사용
                if isinstance(out, (tuple, list)):
                    out = out[0]
                hooks_out[tag] = out.detach()
            return _hook

        hooks_out = {}            # local dict 캡처용

        h1 = module1.register_forward_hook(make_hook("o"))
        h2 = module2.register_forward_hook(make_hook("p"))

        # 한 번만 실행
        with torch.no_grad():
            _ = orig(**dummy_in)
            _ = perm(**dummy_in)

        h1.remove(); h2.remove()

        diff = (hooks_out["o"] - hooks_out["p"]).abs().max().item()
        if diff > EPS:
            mismatch[full_name] = diff
            print(f"[❌] mismatch @ {full_name:60s}  diff={diff:.6f}")
            # 최초 불일치 찾았으면 바로 종료해도 됨
            break
        else:
            print(f"[✅] {full_name:60s}")

        # 재귀 탐색
        register_hooks(module1, module2, full_name)

register_hooks(orig, perm)

if mismatch:
    k, v = next(iter(mismatch.items()))
    print("\n➡️  FIRST layer causing divergence:", k, "| max diff =", v)
else:
    print("\n✅  All layers identical within ±{EPS}")
