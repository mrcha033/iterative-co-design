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

# Debug: Check the actual d_inner dimension in the model
print(f"\n🔍 Analyzing model dimensions:")
for name, mod in perm.named_modules():
    if "MambaMixer" in mod.__class__.__name__:
        print(f"  - {name}: in_proj.out_features = {mod.in_proj.out_features}")
        print(f"    d_inner = {mod.in_proj.out_features // 2}")
        print(f"    out_proj.in_features = {mod.out_proj.in_features}")
        break

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

print(f"\n📊 Permutation tensor length: {len(perm_tensor)}")

# 3) hook 방식 대신 직접 Mixer 출력 비교 ------------------------------------
def check_mixer_equivalence(original_backbone, permuted_backbone):
    """
    Compares the output of each MambaMixer block directly to find the
    first point of divergence with high precision.
    """
    device = next(original_backbone.parameters()).device
    
    orig_mixers = {name: mod for name, mod in original_backbone.named_modules() if "MambaMixer" in mod.__class__.__name__}
    perm_mixers = {name: mod for name, mod in permuted_backbone.named_modules() if "MambaMixer" in mod.__class__.__name__}

    print("\n🔍 Verifying output of each Mixer block...")
    
    mismatch_found = False
    for name, orig_mixer in orig_mixers.items():
        if mismatch_found: break
        
        perm_mixer = perm_mixers[name]
        
        # Create a random input matching the mixer's input dimension (d_model)
        try:
            # Most reliable way to get d_model is from a layer that maps from it
            d_model = orig_mixer.in_proj.in_features
        except AttributeError:
             print(f"[⚠️] Could not determine d_model for {name}, skipping.")
             continue

        x = torch.randn(2, 16, d_model, device=device) # (B, L, D)

        with torch.no_grad():
            y_orig = orig_mixer(x.clone())
            y_perm = perm_mixer(x.clone())
        
        # MambaMixer output can be a tensor or a tuple, handle both
        if isinstance(y_orig, tuple): y_orig = y_orig[0]
        if isinstance(y_perm, tuple): y_perm = y_perm[0]
            
        diff = (y_orig - y_perm).abs().max().item()
        
        if diff > 1e-4:
            print(f"[❌] Mismatch @ {name:40s} | max_diff = {diff:.6f}")
            mismatch_found = True
        else:
            print(f"[✅] {name:40s} | max_diff = {diff:.6f}")

# 4) 최종 모델 전체 출력 비교 (Sanity check) -------------------------------
with torch.no_grad():
    orig_logits = orig(**dummy_in).logits
    perm_logits = perm(**dummy_in).logits
    final_diff = (orig_logits - perm_logits).abs().max().item()

print("\n" + "="*50)
check_mixer_equivalence(orig.backbone, perm.backbone)
print("="*50)

print(f"\nFINAL LOGITS MAX DIFF: {final_diff:.6f}")
if final_diff < 1e-4:
    print("\n✅  SUCCESS: Models are functionally equivalent.")
else:
    print("\n❌  FAILURE: Models are NOT functionally equivalent.")
