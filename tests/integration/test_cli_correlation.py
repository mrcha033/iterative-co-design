"""
Integration tests for correlation matrix CLI script.
"""
import pytest
import tempfile
import subprocess
import sys
from pathlib import Path


class TestCorrelationCLI:
    """Test the correlation matrix generation CLI script."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.script_path = Path(__file__).parent.parent.parent / 'scripts' / 'generate_correlation_matrix.py'
    
    def test_cli_help(self):
        """Test CLI help output."""
        result = subprocess.run([
            sys.executable, str(self.script_path), '--help'
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        assert 'Generate correlation matrices' in result.stdout
        assert '--model' in result.stdout
        assert '--layer' in result.stdout
        assert '--dataset' in result.stdout
    
    def test_cli_cache_info(self):
        """Test CLI cache info command."""
        result = subprocess.run([
            sys.executable, str(self.script_path),
            '--cache-info',
            '--output-dir', self.temp_dir
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        assert 'Cache Information' in result.stdout
        assert 'cached_matrices: 0' in result.stdout or 'Cached matrices: 0' in result.stdout
    
    def test_cli_validation(self):
        """Test CLI input validation."""
        # Test missing required arguments
        result = subprocess.run([
            sys.executable, str(self.script_path),
            '--model', 'invalid-model'
        ], capture_output=True, text=True)
        
        assert result.returncode != 0
        
        # Test invalid model
        result = subprocess.run([
            sys.executable, str(self.script_path),
            '--model', 'invalid-model',
            '--layer', 'test-layer',
            '--dataset', 'wikitext-103',
            '--output-dir', self.temp_dir
        ], capture_output=True, text=True)
        
        assert result.returncode == 1
        assert 'not supported' in result.stdout
    
    @pytest.mark.slow
    def test_cli_with_gcn_model(self):
        """Test CLI with GCN model (if available)."""
        result = subprocess.run([
            sys.executable, str(self.script_path),
            '--model', 'gcn',
            '--layer', 'layers.0',
            '--dataset', 'ogbn-arxiv',
            '--num-samples', '10',
            '--device', 'cpu',
            '--output-dir', self.temp_dir,
            '--force'
        ], capture_output=True, text=True, timeout=60)
        
        # The command might fail due to missing dependencies, but should not crash
        if result.returncode == 0:
            assert 'generated successfully' in result.stdout
        else:
            # Check if it's a known dependency issue
            assert ('torch_geometric' in result.stdout or 
                   'not found' in result.stdout or
                   'not supported' in result.stdout)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])