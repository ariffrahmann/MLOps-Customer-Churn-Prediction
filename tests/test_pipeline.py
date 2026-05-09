import os
import sys
import json
import pytest


class TestDataIntegrity:
    """Test untuk memastikan data tersedia dan valid."""

    def test_data_processed_directory_exists(self):
        assert os.path.exists("data/processed") or os.path.exists("data")

    def test_requirements_file_exists(self):
        assert os.path.exists("requirements.txt")


class TestSourceCode:
    """Test untuk memastikan kode sumber dapat diimpor dengan benar."""

    def test_train_script_exists(self):
        assert os.path.exists("src/modeling/train.py")

    def test_train_script_syntax_valid(self):
        train_path = "src/modeling/train.py"
        if not os.path.exists(train_path):
            pytest.skip("train.py tidak ditemukan")
        
        try:
            with open(train_path, "r") as f:
                code = f.read()
            compile(code, train_path, "exec")
        except SyntaxError as e:
            pytest.fail(f"Syntax error di {train_path}: {e}")

    def test_train_script_has_required_functions(self):
        train_path = "src/modeling/train.py"
        if not os.path.exists(train_path):
            pytest.skip("train.py tidak ditemukan")
        
        with open(train_path, "r") as f:
            code = f.read()

        required_items = ["def load_data", "def run_experiment", "mlflow"]
        for item in required_items:
            assert item in code, f"Item '{item}' harus ada di train.py"


class TestModelArtifacts:
    """Test untuk memvalidasi format output model setelah training."""

    def test_metrics_format(self):
        if not os.path.exists("metrics.json"):
            pytest.skip("metrics.json belum dibuat (akan dibuat saat training)")
        
        with open("metrics.json", "r") as f:
            metrics = json.load(f)
        
        # Pastikan ada minimal accuracy dan f1_score
        required_keys = ["accuracy", "f1_score"]
        for key in required_keys:
            assert key in metrics, f"Metric '{key}' harus ada di metrics.json"
            assert isinstance(metrics[key], (int, float)), \
                f"Metric '{key}' harus berupa angka"
            assert 0 <= metrics[key] <= 1, \
                f"Metric '{key}' harus dalam rentang [0, 1]"


class TestPipelineConfig:
    """Test untuk memvalidasi konfigurasi pipeline."""

    def test_workflow_file_exists(self):
        assert os.path.exists(".github/workflows/mlops-automation.yaml")

    def test_evaluate_script_exists(self):
        assert os.path.exists("scripts/evaluate_model.py")

    def test_register_script_exists(self):
        assert os.path.exists("scripts/register_model.py")


def test_smoke():
    assert 1 + 1 == 2, "Basic math harus benar"


def test_python_version():
    assert sys.version_info >= (3, 8), \
        f"Python harus 3.8+, ditemukan: {sys.version_info}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])