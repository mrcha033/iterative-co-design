#!/usr/bin/env python3
"""
Syntax check for the implemented modules.
"""
import ast
import sys
from pathlib import Path

def check_syntax(file_path):
    """Check syntax of a Python file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        ast.parse(content)
        print(f"✓ {file_path}: Syntax OK")
        return True
    except SyntaxError as e:
        print(f"❌ {file_path}: Syntax Error - {e}")
        return False
    except Exception as e:
        print(f"❌ {file_path}: Error - {e}")
        return False

def main():
    """Check syntax of all implemented modules."""
    print("Checking syntax of implemented modules...")
    print("=" * 50)
    
    # Files to check
    files_to_check = [
        'src/co_design/spectral.py',
        'src/co_design/apply.py',
        'src/co_design/iasp.py',
        'tests/unit/test_spectral.py',
        'tests/unit/test_apply.py',
        'tests/integration/test_permutation_integration.py'
    ]
    
    all_ok = True
    
    for file_path in files_to_check:
        if Path(file_path).exists():
            if not check_syntax(file_path):
                all_ok = False
        else:
            print(f"❌ {file_path}: File not found")
            all_ok = False
    
    print("=" * 50)
    
    if all_ok:
        print("✅ All syntax checks passed!")
        
        # Additional checks
        print("\nChecking module structure...")
        
        # Check imports structure
        try:
            with open('src/co_design/spectral.py', 'r') as f:
                content = f.read()
                assert 'class SpectralClusteringOptimizer' in content
                assert 'def compute_permutation' in content
                assert 'def _construct_affinity_matrix' in content
                assert 'def _compute_graph_laplacian' in content
                assert 'def _compute_eigenvectors' in content
                assert 'def _perform_clustering' in content
                assert 'def _construct_permutation' in content
                print("✓ SpectralClusteringOptimizer: All required methods present")
        except Exception as e:
            print(f"❌ SpectralClusteringOptimizer structure check failed: {e}")
            all_ok = False
        
        try:
            with open('src/co_design/apply.py', 'r') as f:
                content = f.read()
                assert 'class PermutationApplicator' in content
                assert 'def apply_permutation' in content
                assert 'def _validate_permutation' in content
                assert 'def _apply_linear_permutation' in content
                assert 'def _apply_conv1d_permutation' in content
                assert 'def _apply_conv2d_permutation' in content
                print("✓ PermutationApplicator: All required methods present")
        except Exception as e:
            print(f"❌ PermutationApplicator structure check failed: {e}")
            all_ok = False
        
        if all_ok:
            print("✅ All structure checks passed!")
        else:
            print("❌ Some structure checks failed!")
    else:
        print("❌ Some syntax checks failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()