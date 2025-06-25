#!/usr/bin/env python3
import subprocess

# Test if NCU is accessible
def test_ncu():
    print("Testing NCU availability...")
    
    # Check NCU version
    try:
        result = subprocess.run(["ncu", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ NCU found: {result.stdout.strip()}")
        else:
            print(f"✗ NCU error: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ NCU not found: {e}")
        return False
    
    # Test simple CUDA kernel profiling
    print("\nTesting NCU profiling of CUDA kernels...")
    
    # Create a simple test script
    test_script = """
import torch

# Simple CUDA operation
if torch.cuda.is_available():
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.randn(1000, 1000, device='cuda')
    z = torch.matmul(x, y)
    torch.cuda.synchronize()
    print("CUDA operation completed")
else:
    print("CUDA not available")
"""
    
    with open("test_cuda.py", "w") as f:
        f.write(test_script)
    
    # Run NCU on the test script
    cmd = [
        "ncu",
        "--metrics", "l2_cache_hit_rate",
        "--csv",
        "--target-processes", "all",
        "python3", "test_cuda.py"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        print(f"\nNCU exit code: {result.returncode}")
        print(f"NCU stdout length: {len(result.stdout)} chars")
        print(f"NCU stderr length: {len(result.stderr)} chars")
        
        if result.stdout:
            print("\nNCU output (first 500 chars):")
            print(result.stdout[:500])
        
        if result.stderr:
            print("\nNCU errors (first 500 chars):")
            print(result.stderr[:500])
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"\nNCU profiling failed: {e}")
        return False
    finally:
        # Cleanup
        import os
        if os.path.exists("test_cuda.py"):
            os.remove("test_cuda.py")

if __name__ == "__main__":
    success = test_ncu()
    print(f"\nNCU test {'PASSED' if success else 'FAILED'}") 