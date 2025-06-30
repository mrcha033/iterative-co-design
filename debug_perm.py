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

# 3) hook 으로 층별 출력 비교 (효율적인 방식) ---------------------------------
EPS = 1e-3  # Relaxed epsilon for numerical stability with fp16/fp32 conversions
mismatch = OrderedDict()

def extract_tensor(output):
    """Recursively extracts a tensor from various model output types."""
    if torch.is_tensor(output):
        return output
    if is_dataclass(output):
        # Prefer 'last_hidden_state', fallback to the last item in 'hidden_states'
        if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
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

def collect_all_outputs(model, model_inputs):
    """
    Attaches hooks to all submodules, runs a single forward pass,
    and collects all intermediate outputs in a dictionary.
    """
    outputs = {}
    hooks = []

    def make_hook(name):
        def _hook(mod, inp, out):
            try:
                outputs[name] = extract_tensor(out).detach()
            except (TypeError, AttributeError) as e:
                print(f"[⚠️] Hook error at {name} with output type {type(out)}: {e}")
                outputs[name] = None
        return _hook

    for name, module in model.named_modules():
        hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        _ = model(**model_inputs)

    for h in hooks:
        h.remove()
    
    return outputs

print("\nCollecting outputs from original model...")
orig_outputs = collect_all_outputs(orig, dummy_in)

print("Collecting outputs from permuted model...")
perm_outputs = collect_all_outputs(perm, dummy_in)

print("\nComparing outputs...")
# Use sorted keys to ensure consistent comparison order and parent-first traversal
sorted_keys = sorted(orig_outputs.keys(), key=lambda x: (x.count('.'), x))

for name in sorted_keys:
    o_out = orig_outputs.get(name)
    p_out = perm_outputs.get(name)

    if o_out is None or p_out is None:
        # A hook error was already printed, so just skip.
        continue
        
    if o_out.shape != p_out.shape:
        print(f"[❌] Mismatch @ {name:60s}  SHAPE DIFFERENCE! orig: {o_out.shape}, perm: {p_out.shape}")
        mismatch[name] = float('inf')
        break

    diff = (o_out - p_out).abs().max().item()
    if diff > EPS:
        mismatch[name] = diff
        print(f"[❌] Mismatch @ {name:60s}  diff={diff:.6f}")
        # Break after finding the first divergence to pinpoint the root cause.
        break
    elif name.count('.') < 2: # Print success only for major blocks to reduce noise
        print(f"[✅] {name:60s}  (diff: {diff:.6f})")

if mismatch:
    k, v = next(iter(mismatch.items()))
    print(f"\n➡️  FIRST layer causing divergence: {k} | max diff = {v}")
else:
    print(f"\n✅  All layers are functionally equivalent (diff <= {EPS}).")
