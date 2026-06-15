"""Unit tests for controller/classifiers.py (the shipped WorkloadClassifiers)."""
from pathlib import Path

import pytest

from code_capsules.api import TaskDescriptor
from code_capsules.controller.classifiers import (
    GenericPromptClassifier,
    YamlMappingClassifier,
    BenchmarkSaturatedClassifier,
    SHIPPED_REPO_CLASSES,
)


def _task(text="", **ctx) -> TaskDescriptor:
    return TaskDescriptor(text=text, context=ctx)


class TestGenericPromptClassifier:
    def test_named_when_file_mentioned(self):
        c = GenericPromptClassifier()
        assert c.classify(_task("the bug is in foo/bar.py at line 10")) == "find_named"

    def test_symptom_when_behavioural_only(self):
        c = GenericPromptClassifier()
        assert c.classify(_task("the output is wrong when I sort large lists")) == "find_symptom"


class TestBenchmarkSaturatedClassifier:
    def test_always_saturated(self):
        c = BenchmarkSaturatedClassifier()
        assert c.classify(_task("anything at all")) == "saturated_workload"


class TestYamlMappingClassifier:
    def test_shipped_table_exists(self):
        assert Path(SHIPPED_REPO_CLASSES).exists()

    def test_default_constructs_with_literal_name(self):
        # Zero-arg => loads the shipped SWE-bench table, name is the literal
        # "yaml_mapping" (the paper's Table-2 selector name).
        c = YamlMappingClassifier()
        assert c.name == "yaml_mapping"

    @pytest.mark.parametrize("repo,expected", [
        ("sympy", "scientific_workload"),
        ("scikit-learn", "scientific_workload"),
        ("matplotlib", "visualization_workload"),
        ("django", "hard_workload"),
    ])
    def test_classifies_shipped_repos(self, repo, expected):
        assert YamlMappingClassifier().classify(_task(repo=repo)) == expected

    def test_unknown_repo_falls_back_to_default(self):
        assert YamlMappingClassifier().classify(_task(repo="acme-corp")) == "hard_workload"

    def test_missing_key_field_falls_back_to_default(self):
        assert YamlMappingClassifier().classify(_task()) == "hard_workload"

    def test_custom_table_keeps_distinct_registry_name(self, tmp_path):
        p = tmp_path / "domains.yaml"
        p.write_text("key_field: lang\ndefault: other\nmapping:\n  rust: systems\n")
        c = YamlMappingClassifier(p)
        assert c.name == "yaml_mapping:domains"          # distinct so tables coexist
        assert c.classify(_task(lang="rust")) == "systems"
        assert c.classify(_task(lang="cobol")) == "other"

    def test_explicit_name_override(self, tmp_path):
        p = tmp_path / "t.yaml"
        p.write_text("mapping: {}\n")
        assert YamlMappingClassifier(p, name="my_map").name == "my_map"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            YamlMappingClassifier(tmp_path / "nope.yaml")
