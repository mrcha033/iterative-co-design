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
from dataclasses import is_dataclass
from torch.utils.data import DataLoader, Dataset

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

# --- Create a proper dataloader with diverse data for correlation analysis ---
class RandomTextDataset(Dataset):
    """Generates random token sequences to ensure activation variance."""
    def __init__(self, tokenizer, num_samples=128, seq_len=128):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = tokenizer.vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

# Use a real dataloader to get varied activations
random_dataset = RandomTextDataset(tok, num_samples=64, seq_len=128)
data_loader = DataLoader(random_dataset, batch_size=8)

# 2) IASP 실행  --------------------------------------------------------------
#   이제 충분한 샘플로 correlation을 계산합니다.
perm_tensor, _ = run_iasp_on_mamba(
    perm,
    dataloader=data_loader,
    iasp_config=OmegaConf.create({
        "max_samples": 4096,                  # Use enough samples for stable correlation
        "cluster_size_range": [16, 64],       # 빠른 실행용
        "knn_k": 64,
        "spectral_n_init": 10,
        "spectral_random_state": 42
    }),
)

# 3) hook 으로 층별 출력 비교 -----------------------------------------------
EPS = 1e-4
mismatch = OrderedDict()

def extract_tensor(output):
    """Recursively extracts a tensor from various model output types."""
    if torch.is_tensor(output):
        return output
    if is_dataclass(output):
        # Prefer 'last_hidden_state', fallback to the last item in 'hidden_states'
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        if hasattr(output, "hidden_states") and output.hidden_states:
            return output.hidden_states[-1]
    if isinstance(output, (list, tuple)) and output:
        return extract_tensor(output[0])
    if isinstance(output, dict):
        # Common keys in HF model outputs
        for key in ["last_hidden_state", "logits", "hidden_states"]:
            if key in output:
                return extract_tensor(output[key])
    raise TypeError(f"Unsupported or empty output type: {type(output)}")

def register_hooks(m1, m2, prefix=""):
    for (name1, module1), (name2, module2) in zip_longest(
        m1.named_children(), m2.named_children()
    ):
        # 서브 모듈이 없으면 leaf → 비교
        if module1 is None or module2 is None:
            continue
        full_name = f"{prefix}.{name1}" if prefix else name1

        def make_hook(storage, tag):
            def _hook(mod, inp, out):
                try:
                    tensor_out = extract_tensor(out)
                    storage[tag] = tensor_out.detach()
                except (TypeError, AttributeError) as e:
                    print(f"Hook error at {full_name} with output type {type(out)}: {e}")
                    storage[tag] = None
            return _hook

        hooks_out = {} # Use a new dict for each recursive call's scope

        h1 = module1.register_forward_hook(make_hook(hooks_out, "o"))
        h2 = module2.register_forward_hook(make_hook(hooks_out, "p"))

        # 한 번만 실행
        with torch.no_grad():
            _ = orig(**dummy_in)
            _ = perm(**dummy_in)

        h1.remove(); h2.remove()

        if hooks_out.get("o") is None or hooks_out.get("p") is None:
            print(f"[⚠️] Could not compare outputs for {full_name}. Skipping.")
            continue

        diff = (hooks_out["o"] - hooks_out["p"]).abs().max().item()
        if diff > EPS:
            mismatch[full_name] = diff
            print(f"[❌] mismatch @ {full_name:60s}  diff={diff:.6f}")
            # 최초 불일치 찾았으면 바로 종료해도 됨
            # For full debug, comment out the break
            # break 
        else:
            print(f"[✅] {full_name:60s}")

        # 재귀 탐색
        if not mismatch: # Stop recursion if a mismatch is found
            register_hooks(module1, module2, full_name)

register_hooks(orig.backbone, perm.backbone) # Compare backbone modules where changes occur

if mismatch:
    k, v = next(iter(mismatch.items()))
    print("\n➡️  FIRST layer causing divergence:", k, "| max diff =", v)
else:
    print("\n✅  All layers identical within ±{EPS}")
