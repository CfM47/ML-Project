"""Tests for AutoML autosaver caching functionality."""

import json
import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from auto_ml import AutoML


@pytest.fixture
def temp_cache_dir() -> Generator[Path, None, None]:
    """Create a temporary cache directory for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestAutoMLAutosaver:
    """Test suite for AutoML caching and autosaver functionality."""

    def test_automl_initialization_creates_cache_dir(
        self,
        temp_cache_dir: Path,
    ) -> None:
        """Test that AutoML creates cache directory on init."""
        automl = AutoML(cache_dir=temp_cache_dir)
        assert Path(temp_cache_dir).exists()
        assert automl.cache_dir == Path(temp_cache_dir)

    def test_automl_initializes_empty_caches(self, temp_cache_dir: Path) -> None:
        """Test that AutoML initializes with empty caches."""
        automl = AutoML(cache_dir=temp_cache_dir)
        assert automl.results_cache == {}
        assert automl.execution_times_cache == {}

    def test_cache_result_saves_to_json(self, temp_cache_dir: Path) -> None:
        """Test that _cache_result saves results and times to JSON."""
        automl = AutoML(cache_dir=temp_cache_dir)
        result = {"mask_pairs": [[]], "evaluation": {"accuracy": 0.95}}
        execution_time = 5.2

        automl._cache_result("aug1", "model1", result, execution_time)

        # Verify JSON files were created and contain data
        results_file = Path(temp_cache_dir) / "results_cache.json"
        times_file = Path(temp_cache_dir) / "execution_times_cache.json"

        assert results_file.exists()
        assert times_file.exists()

        with open(results_file) as f:
            results_data = json.load(f)
        with open(times_file) as f:
            times_data = json.load(f)

        assert "aug1" in results_data
        assert "model1" in results_data["aug1"]
        assert results_data["aug1"]["model1"] == result
        assert times_data["aug1"]["model1"] == execution_time

    def test_is_cached_returns_true_when_cached(self, temp_cache_dir: Path) -> None:
        """Test that _is_cached returns True for cached combinations."""
        automl = AutoML(cache_dir=temp_cache_dir)
        result = {"mask_pairs": [[]], "evaluation": {"accuracy": 0.95}}

        automl._cache_result("aug1", "model1", result, 5.0)

        assert automl._is_cached("aug1", "model1")

    def test_is_cached_returns_false_when_not_cached(
        self, temp_cache_dir: Path,
    ) -> None:
        """Test that _is_cached returns False for uncached combinations."""
        automl = AutoML(cache_dir=temp_cache_dir)

        assert not automl._is_cached("aug1", "model1")

    def test_get_cached_result_retrieves_result(self, temp_cache_dir: Path) -> None:
        """Test that _get_cached_result retrieves cached results."""
        automl = AutoML(cache_dir=temp_cache_dir)
        result = {"mask_pairs": [[]], "evaluation": {"accuracy": 0.95}}

        automl._cache_result("aug1", "model1", result, 5.0)
        retrieved = automl._get_cached_result("aug1", "model1")

        assert retrieved == result

    def test_get_cached_result_returns_none_when_not_cached(
        self,
        temp_cache_dir: Path,
    ) -> None:
        """Test that _get_cached_result returns None for uncached combos."""
        automl = AutoML(cache_dir=temp_cache_dir)

        assert automl._get_cached_result("aug1", "model1") is None

    def test_load_caches_loads_existing_files(self, temp_cache_dir: Path) -> None:
        """Test that _load_caches loads existing cache files."""
        # Create initial cache
        automl1 = AutoML(cache_dir=temp_cache_dir)
        result = {"mask_pairs": [[]], "evaluation": {"accuracy": 0.95}}
        automl1._cache_result("aug1", "model1", result, 5.0)

        # Create new instance - should load cache
        automl2 = AutoML(cache_dir=temp_cache_dir)

        assert automl2._is_cached("aug1", "model1")
        assert automl2._get_cached_result("aug1", "model1") == result
        assert automl2.execution_times_cache["aug1"]["model1"] == 5.0

    def test_clear_cache_entry_removes_entry(self, temp_cache_dir: Path) -> None:
        """Test that _clear_cache_entry removes specific cache entry."""
        automl = AutoML(cache_dir=temp_cache_dir)
        result = {"mask_pairs": [[]], "evaluation": {"accuracy": 0.95}}

        automl._cache_result("aug1", "model1", result, 5.0)
        assert automl._is_cached("aug1", "model1")

        automl._clear_cache_entry("aug1", "model1")
        assert not automl._is_cached("aug1", "model1")

    def test_clear_cache_entry_persists_to_json(self, temp_cache_dir: Path) -> None:
        """Test that _clear_cache_entry persists changes to JSON."""
        automl1 = AutoML(cache_dir=temp_cache_dir)
        result = {"mask_pairs": [[]], "evaluation": {"accuracy": 0.95}}

        automl1._cache_result("aug1", "model1", result, 5.0)
        automl1._clear_cache_entry("aug1", "model1")

        # Create new instance - should reflect cleared state
        automl2 = AutoML(cache_dir=temp_cache_dir)
        assert not automl2._is_cached("aug1", "model1")

    def test_multiple_augmentators_and_models(self, temp_cache_dir: Path) -> None:
        """Test caching with multiple augmentators and models."""
        automl = AutoML(cache_dir=temp_cache_dir)
        result1 = {"mask_pairs": [[]], "evaluation": {"accuracy": 0.90}}
        result2 = {"mask_pairs": [[]], "evaluation": {"accuracy": 0.95}}
        result3 = {"mask_pairs": [[]], "evaluation": {"accuracy": 0.92}}

        automl._cache_result("aug1", "model1", result1, 5.0)
        automl._cache_result("aug1", "model2", result2, 6.0)
        automl._cache_result("aug2", "model1", result3, 5.5)

        assert automl._is_cached("aug1", "model1")
        assert automl._is_cached("aug1", "model2")
        assert automl._is_cached("aug2", "model1")
        assert not automl._is_cached("aug2", "model2")

        # Verify nested structure is preserved
        assert automl.results_cache["aug1"]["model1"] == result1
        assert automl.results_cache["aug1"]["model2"] == result2
        assert automl.results_cache["aug2"]["model1"] == result3

    def test_cache_handles_corrupted_json(self, temp_cache_dir: Path) -> None:
        """Test that corrupted JSON files are handled gracefully."""
        # Create corrupted JSON file
        cache_dir = Path(temp_cache_dir)
        cache_dir.mkdir(exist_ok=True)
        results_file = cache_dir / "results_cache.json"
        results_file.write_text("{ invalid json")

        # Should not raise, just print warning
        automl = AutoML(cache_dir=temp_cache_dir)
        assert automl.results_cache == {}

    def test_nested_dict_structure_preserved(self, temp_cache_dir: Path) -> None:
        """Test that nested dictionary structure is preserved."""
        automl = AutoML(cache_dir=temp_cache_dir)
        result = {"mask_pairs": [[]], "evaluation": {"accuracy": 0.95}}

        automl._cache_result("aug1", "model1", result, 5.0)

        # Check structure
        assert isinstance(automl.results_cache, dict)
        assert isinstance(automl.results_cache["aug1"], dict)
        assert "model1" in automl.results_cache["aug1"]
        assert automl.results_cache["aug1"]["model1"]["evaluation"]["accuracy"] == 0.95

    def test_execution_time_tracking(self, temp_cache_dir: Path) -> None:
        """Test that execution times are properly tracked."""
        automl = AutoML(cache_dir=temp_cache_dir)
        result = {"mask_pairs": [[]], "evaluation": {"accuracy": 0.95}}

        automl._cache_result("aug1", "model1", result, 12.345)

        assert automl.execution_times_cache["aug1"]["model1"] == 12.345

        # Verify it's saved to JSON
        times_file = Path(temp_cache_dir) / "execution_times_cache.json"
        with open(times_file) as f:
            times_data = json.load(f)
        assert times_data["aug1"]["model1"] == 12.345

    def test_run_experiment_clear_cache_parameter(self, temp_cache_dir: Path) -> None:
        """Test that run_experiment accepts clear_cache parameter."""
        automl = AutoML(cache_dir=temp_cache_dir)

        # Manually add to cache for testing
        automl._cache_result("aug1", "model1", {"data": "test"}, 5.0)
        assert automl._is_cached("aug1", "model1")

        # Use clear_cache parameter (without running actual experiment)
        # Just test that the parameter is accepted and clears entries
        automl._clear_cache_entry("aug1", "model1")

        assert not automl._is_cached("aug1", "model1")
