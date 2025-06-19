import pytest
import torch
import torch.nn as nn
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from utils.profiler import LatencyProfiler


class SimpleTestModel(nn.Module):
    """Simple model for testing profiler functionality."""

    def __init__(self, d_model=32):
        super().__init__()
        self.linear = nn.Linear(d_model, 10)

    def forward(self, input_ids):
        return {"logits": self.linear(input_ids.float())}


class TestLatencyProfiler:
    """Unit tests for LatencyProfiler class."""

    def test_profiler_initialization(self):
        """Test LatencyProfiler initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            profiler = LatencyProfiler(cache_dir=temp_dir)
            assert profiler.cache_dir == Path(temp_dir)
            assert profiler.ncu_metrics == ["l2_tex_hit_rate.pct"]
            assert profiler.cache_file == Path(temp_dir) / "profiler_cache.json"

    def test_model_hash_generation(self):
        """Test that model hash generation works consistently."""
        profiler = LatencyProfiler()
        model1 = SimpleTestModel()
        model2 = SimpleTestModel()

        # Set same random seed for reproducible weights
        torch.manual_seed(42)
        model1 = SimpleTestModel()
        torch.manual_seed(42)
        model2 = SimpleTestModel()

        # Same architecture and weights should give same hash
        hash1 = profiler._get_model_hash(model1.state_dict())
        hash2 = profiler._get_model_hash(model2.state_dict())
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hash length

        # Different weights should give different hash
        model1.linear.weight.data.fill_(1.0)
        hash3 = profiler._get_model_hash(model1.state_dict())
        assert hash3 != hash1

    def test_model_hash_deterministic(self):
        """Test that model hashing is deterministic across multiple calls."""
        # Create two identical models
        model1 = SimpleTestModel(4)
        model2 = SimpleTestModel(4)

        # Set identical weights
        torch.manual_seed(42)
        with torch.no_grad():
            for param in model1.parameters():
                param.fill_(1.0)

        with torch.no_grad():
            for param in model2.parameters():
                param.fill_(1.0)

        profiler = LatencyProfiler()

        # Generate hashes multiple times
        hash1_first = profiler._get_model_hash(model1.state_dict())
        hash1_second = profiler._get_model_hash(model1.state_dict())
        hash2 = profiler._get_model_hash(model2.state_dict())

        # All hashes should be identical for identical models
        assert hash1_first == hash1_second, (
            "Hash should be deterministic for same model"
        )
        assert hash1_first == hash2, (
            "Hash should be identical for models with same weights"
        )

        # Modify one model slightly
        with torch.no_grad():
            list(model2.parameters())[0][0][0] = 2.0

        hash2_modified = profiler._get_model_hash(model2.state_dict())

        # Hash should change when model changes
        assert hash1_first != hash2_modified, (
            "Hash should change when model weights change"
        )

    def test_cache_read_write(self):
        """Test cache read/write functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            profiler = LatencyProfiler(cache_dir=temp_dir)

            # Test reading empty cache
            cache = profiler._read_cache()
            assert cache == {}

            # Test writing and reading cache
            test_data = {"model_hash_1": {"metric1": 0.85, "metric2": 0.92}}
            profiler._write_cache(test_data)

            cache = profiler._read_cache()
            assert cache == test_data

    def test_cache_read_invalid_json(self):
        """Test cache read with invalid JSON."""
        with tempfile.TemporaryDirectory() as temp_dir:
            profiler = LatencyProfiler(cache_dir=temp_dir)

            # Write invalid JSON
            with open(profiler.cache_file, "w") as f:
                f.write("invalid json content")

            # Should return empty dict for invalid JSON
            cache = profiler._read_cache()
            assert cache == {}

    def test_measure_latency_cpu(self):
        """Test CPU latency measurement."""
        profiler = LatencyProfiler()
        model = SimpleTestModel(d_model=32)
        model.eval()

        dummy_input = {
            "input_ids": torch.randn(2, 10, 32)
        }  # Proper shape: (batch, seq, features)

        latency = profiler.measure_latency(model, dummy_input, num_runs=5)

        assert isinstance(latency, float)
        assert latency > 0  # Should take some time
        assert latency < 1000  # Should be reasonable for small model

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_measure_latency_gpu(self):
        """Test GPU latency measurement."""
        profiler = LatencyProfiler()
        model = SimpleTestModel().cuda()
        model.eval()

        dummy_input = {"input_ids": torch.randint(0, 100, (2, 10))}

        latency = profiler.measure_latency(model, dummy_input, num_runs=5)

        assert isinstance(latency, float)
        assert latency > 0

    def test_measure_latency_device_agnostic(self):
        """Test latency measurement works on both CPU and GPU when available."""
        profiler = LatencyProfiler()

        # Test CPU measurement
        model_cpu = SimpleTestModel(d_model=32)
        model_cpu.eval()
        dummy_input = {"input_ids": torch.randn(2, 10, 32)}

        latency_cpu = profiler.measure_latency(model_cpu, dummy_input, num_runs=3)

        assert isinstance(latency_cpu, float)
        assert latency_cpu > 0
        assert latency_cpu < 1000  # Should be reasonable

        # Test GPU measurement if available
        if torch.cuda.is_available():
            try:
                model_gpu = SimpleTestModel(d_model=32).cuda()
                model_gpu.eval()
                dummy_input_gpu = {"input_ids": torch.randn(2, 10, 32).cuda()}

                latency_gpu = profiler.measure_latency(
                    model_gpu, dummy_input_gpu, num_runs=3
                )

                assert isinstance(latency_gpu, float)
                assert latency_gpu > 0

                # GPU should generally be faster for same operations (though not always guaranteed)
                # Just ensure both measurements are reasonable
                assert latency_gpu < 1000

            except RuntimeError:
                # GPU might have insufficient memory or other issues, just continue
                pass

    def test_measure_cache_hits_no_cuda(self):
        """Test cache measurement when CUDA is not available."""
        profiler = LatencyProfiler()
        model = SimpleTestModel()
        dummy_input = {"input_ids": torch.randint(0, 100, (2, 10))}

        with patch("torch.cuda.is_available", return_value=False):
            result = profiler.measure_cache_hits(model, dummy_input)
            assert result is None

    def test_measure_cache_hits_no_ncu(self):
        """Test cache measurement when NCU is not available."""
        profiler = LatencyProfiler()
        model = SimpleTestModel()
        dummy_input = {"input_ids": torch.randint(0, 100, (2, 10))}

        with patch("torch.cuda.is_available", return_value=True), patch(
            "shutil.which", return_value=None
        ):
            result = profiler.measure_cache_hits(model, dummy_input)
            assert result is None

    def test_measure_cache_hits_cache_hit(self):
        """Test cache measurement with cache hit."""
        with tempfile.TemporaryDirectory() as temp_dir:
            profiler = LatencyProfiler(cache_dir=temp_dir)
            model = SimpleTestModel()
            dummy_input = {"input_ids": torch.randint(0, 100, (2, 10))}

            # Pre-populate cache
            model_hash = profiler._get_model_hash(model.state_dict())
            cached_metrics = {"l2_tex_hit_rate.pct": 85.0}
            profiler._write_cache({model_hash: cached_metrics})

            with patch("torch.cuda.is_available", return_value=True), patch(
                "shutil.which", return_value="/usr/bin/ncu"
            ):
                result = profiler.measure_cache_hits(model, dummy_input)
                assert result == cached_metrics

    @patch("subprocess.run")
    def test_measure_cache_hits_subprocess_success(self, mock_subprocess):
        """Test successful NCU profiling via subprocess."""
        with tempfile.TemporaryDirectory() as temp_dir:
            profiler = LatencyProfiler(cache_dir=temp_dir)
            model = SimpleTestModel()
            dummy_input = {"input_ids": torch.randint(0, 100, (2, 10))}

            # Mock successful subprocess run
            mock_subprocess.return_value = MagicMock(returncode=0)

            # Mock NCU output parsing
            mock_metrics = {"l2_tex_hit_rate.pct": 78.5}

            with patch("torch.cuda.is_available", return_value=True), patch(
                "shutil.which", return_value="/usr/bin/ncu"
            ), patch.object(
                profiler, "_parse_ncu_output", return_value=mock_metrics
            ), patch("torch.save"):  # Mock torch.save to avoid file I/O
                result = profiler.measure_cache_hits(model, dummy_input)
                assert result == mock_metrics

                # Verify subprocess was called with correct arguments
                mock_subprocess.assert_called_once()
                call_args = mock_subprocess.call_args[0][0]
                assert call_args[0] == "ncu"
                assert "--metrics" in call_args
                assert "l2_tex_hit_rate.pct" in call_args

    @patch("subprocess.run")
    def test_measure_cache_hits_subprocess_failure(self, mock_subprocess):
        """Test NCU profiling failure via subprocess."""
        profiler = LatencyProfiler()
        model = SimpleTestModel()
        dummy_input = {"input_ids": torch.randint(0, 100, (2, 10))}

        # Mock failed subprocess run
        from subprocess import CalledProcessError

        mock_subprocess.side_effect = CalledProcessError(1, "ncu", stderr="NCU error")

        with patch("torch.cuda.is_available", return_value=True), patch(
            "shutil.which", return_value="/usr/bin/ncu"
        ), patch("torch.save"):  # Mock torch.save to avoid file I/O
            result = profiler.measure_cache_hits(model, dummy_input)
            assert result is None

    def test_parse_ncu_output_success(self):
        """Test successful NCU output parsing."""
        profiler = LatencyProfiler(
            ncu_metrics=["l2_tex_hit_rate.pct", "sm__cycles_elapsed.avg"]
        )

        # Mock NCU CSV output - adjust pattern to match actual NCU format
        mock_output = """l2_tex_hit_rate.pct,%,85.32
sm__cycles_elapsed.avg,cycle,1234.56"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write(mock_output)
            f.flush()

            result = profiler._parse_ncu_output(Path(f.name))

            assert result is not None
            assert "l2_tex_hit_rate.pct" in result
            assert result["l2_tex_hit_rate.pct"] == 85.32

    def test_parse_ncu_output_failure(self):
        """Test NCU output parsing failure."""
        profiler = LatencyProfiler()

        # Mock invalid output file
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("invalid output format")
            f.flush()

            result = profiler._parse_ncu_output(Path(f.name))
            assert result is None

    def test_parse_ncu_output_missing_file(self):
        """Test NCU output parsing with missing file."""
        profiler = LatencyProfiler()

        result = profiler._parse_ncu_output(Path("/nonexistent/file.txt"))
        assert result is None

    def test_parse_ncu_output_scientific_notation(self):
        """Test parsing of NCU output with scientific notation numbers."""
        profiler = LatencyProfiler(ncu_metrics=["l2_tex_hit_rate.pct", "dram_read_throughput.avg.pct_of_peak_sustained_elapsed"])
        
        # Mock NCU CSV output with scientific notation
        mock_output = """l2_tex_hit_rate.pct,%,1.23e+02
dram_read_throughput.avg.pct_of_peak_sustained_elapsed,%,4.56e-01"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write(mock_output)
            f.flush()
            file_name = f.name

        # Parse after file is closed
        result = profiler._parse_ncu_output(Path(file_name))
        
        # Cleanup after parsing
        Path(file_name).unlink()
        
        # Verify results
        assert result is not None
        assert "l2_tex_hit_rate.pct" in result
        assert "dram_read_throughput.avg.pct_of_peak_sustained_elapsed" in result
        assert abs(result["l2_tex_hit_rate.pct"] - 123.0) < 1e-6
        assert abs(result["dram_read_throughput.avg.pct_of_peak_sustained_elapsed"] - 0.456) < 1e-6

    def test_parse_ncu_output_mixed_notation(self):
        """Test parsing of NCU output with both standard and scientific notation."""
        profiler = LatencyProfiler(ncu_metrics=["l2_tex_hit_rate.pct", "dram_read_throughput.avg.pct_of_peak_sustained_elapsed"])
        
        # Mock NCU CSV output with mixed notation formats
        mock_output = """l2_tex_hit_rate.pct,%,78.45
dram_read_throughput.avg.pct_of_peak_sustained_elapsed,%,1.23E+05"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write(mock_output)
            f.flush()
            file_name = f.name

        # Parse after file is closed
        result = profiler._parse_ncu_output(Path(file_name))
        
        # Cleanup after parsing
        Path(file_name).unlink()
        
        # Verify results
        assert result is not None
        assert "l2_tex_hit_rate.pct" in result
        assert "dram_read_throughput.avg.pct_of_peak_sustained_elapsed" in result
        assert abs(result["l2_tex_hit_rate.pct"] - 78.45) < 1e-6
        assert abs(result["dram_read_throughput.avg.pct_of_peak_sustained_elapsed"] - 123000.0) < 1e-6
