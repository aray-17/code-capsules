"""Unit tests for controller/yaml_dsl.py."""
import textwrap
from pathlib import Path
import pytest
from code_capsules.controller.yaml_dsl import load_policy, PolicyError
from code_capsules.controller.routing_config import RoutingConfig, DEFAULT_CONFIG


# ── Helpers ───────────────────────────────────────────────────────────────────

def write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "policy.yaml"
    p.write_text(textwrap.dedent(content))
    return p


# ── Basic loading ─────────────────────────────────────────────────────────────

class TestLoadPolicy:
    def test_empty_file_returns_defaults(self, tmp_path):
        p = write_yaml(tmp_path, "routing: {}")
        cfg = load_policy(p)
        assert cfg.fine_threshold == DEFAULT_CONFIG.fine_threshold
        assert cfg.compound_threshold == DEFAULT_CONFIG.compound_threshold

    def test_override_fine_threshold(self, tmp_path):
        p = write_yaml(tmp_path, """
            routing:
              fine_threshold: 0.15
        """)
        cfg = load_policy(p)
        assert cfg.fine_threshold == 0.15
        assert cfg.compound_threshold == DEFAULT_CONFIG.compound_threshold

    def test_override_compound_threshold(self, tmp_path):
        p = write_yaml(tmp_path, """
            routing:
              compound_threshold: 0.60
        """)
        cfg = load_policy(p)
        assert cfg.compound_threshold == 0.60

    def test_override_bash_threshold(self, tmp_path):
        p = write_yaml(tmp_path, """
            routing:
              bash_sequential_threshold: 0.30
        """)
        cfg = load_policy(p)
        assert cfg.bash_sequential_threshold == 0.30

    def test_override_par_ratio_threshold(self, tmp_path):
        p = write_yaml(tmp_path, """
            routing:
              par_ratio_threshold: 0.25
        """)
        cfg = load_policy(p)
        assert cfg.par_ratio_threshold == 0.25

    def test_returns_routing_config(self, tmp_path):
        p = write_yaml(tmp_path, "routing: {}")
        cfg = load_policy(p)
        assert isinstance(cfg, RoutingConfig)

    def test_load_from_string(self):
        from code_capsules.controller.yaml_dsl import load_policy_string
        cfg = load_policy_string("routing:\n  fine_threshold: 0.10\n")
        assert cfg.fine_threshold == 0.10


# ── Compound gate configuration ───────────────────────────────────────────────

class TestCompoundGateConfig:
    def test_task_type_whitelist(self, tmp_path):
        p = write_yaml(tmp_path, """
            routing:
              compound_gate:
                task_type_whitelist: [refactor, new_feature, multi_step]
        """)
        cfg = load_policy(p)
        assert "multi_step" in cfg.compound_task_type_whitelist
        assert "refactor" in cfg.compound_task_type_whitelist

    def test_empty_whitelist_allows_all(self, tmp_path):
        p = write_yaml(tmp_path, """
            routing:
              compound_gate:
                task_type_whitelist: []
        """)
        cfg = load_policy(p)
        assert cfg.compound_task_type_whitelist == []

    def test_min_distinct_files(self, tmp_path):
        p = write_yaml(tmp_path, """
            routing:
              compound_gate:
                min_distinct_files: 5
        """)
        cfg = load_policy(p)
        assert cfg.compound_min_distinct_files == 5

    def test_min_rbw_ratio(self, tmp_path):
        p = write_yaml(tmp_path, """
            routing:
              compound_gate:
                min_rbw_ratio: 0.20
        """)
        cfg = load_policy(p)
        assert cfg.compound_min_rbw_ratio == 0.20


# ── Formula weights ───────────────────────────────────────────────────────────

class TestFormulaWeights:
    def test_override_single_weight(self, tmp_path):
        p = write_yaml(tmp_path, """
            routing:
              formula_weights:
                parallelizable_ratio: 0.50
        """)
        cfg = load_policy(p)
        assert cfg.formula_weights["parallelizable_ratio"] == 0.50
        # Unspecified weights keep defaults
        assert cfg.formula_weights["context_load_ratio"] == DEFAULT_CONFIG.formula_weights["context_load_ratio"]

    def test_override_all_weights(self, tmp_path):
        p = write_yaml(tmp_path, """
            routing:
              formula_weights:
                parallelizable_ratio: 0.30
                context_load_ratio: 0.25
                read_before_write_ratio: 0.25
                tool_calls_per_turn: 0.15
                write_concentration: -0.05
        """)
        cfg = load_policy(p)
        assert cfg.formula_weights["parallelizable_ratio"] == 0.30
        assert cfg.formula_weights["write_concentration"] == -0.05

    def test_unknown_weight_key_raises(self, tmp_path):
        p = write_yaml(tmp_path, """
            routing:
              formula_weights:
                nonexistent_feature: 0.99
        """)
        with pytest.raises(PolicyError, match="unknown weight"):
            load_policy(p)


# ── Quality gate ──────────────────────────────────────────────────────────────

class TestQualityGateConfig:
    def test_lint_weight(self, tmp_path):
        p = write_yaml(tmp_path, """
            routing:
              quality_gate:
                lint_weight: 0.20
        """)
        cfg = load_policy(p)
        assert cfg.quality_lint_weight == 0.20

    def test_gate_mode_binary(self, tmp_path):
        p = write_yaml(tmp_path, """
            routing:
              quality_gate:
                mode: binary
        """)
        cfg = load_policy(p)
        assert cfg.quality_gate_mode == "binary"

    def test_gate_mode_python(self, tmp_path):
        p = write_yaml(tmp_path, """
            routing:
              quality_gate:
                mode: python
        """)
        cfg = load_policy(p)
        assert cfg.quality_gate_mode == "python"

    def test_invalid_gate_mode_raises(self, tmp_path):
        p = write_yaml(tmp_path, """
            routing:
              quality_gate:
                mode: cobol
        """)
        with pytest.raises(PolicyError, match="quality_gate.mode"):
            load_policy(p)


# ── Scope keywords ────────────────────────────────────────────────────────────

class TestScopeKeywords:
    def test_override_scope_keywords(self, tmp_path):
        p = write_yaml(tmp_path, """
            routing:
              scope_keywords:
                - all documents
                - entire corpus
        """)
        cfg = load_policy(p)
        assert "all documents" in cfg.scope_keywords
        assert "entire corpus" in cfg.scope_keywords
        # Coding defaults replaced, not merged
        assert "across all" not in cfg.scope_keywords

    def test_extend_scope_keywords(self, tmp_path):
        p = write_yaml(tmp_path, """
            routing:
              scope_keywords_extra:
                - all documents
                - entire corpus
        """)
        cfg = load_policy(p)
        # Extra keywords merged with defaults
        assert "across all" in cfg.scope_keywords
        assert "all documents" in cfg.scope_keywords


# ── Validation ────────────────────────────────────────────────────────────────

class TestValidation:
    def test_fine_above_compound_raises(self, tmp_path):
        p = write_yaml(tmp_path, """
            routing:
              fine_threshold: 0.60
              compound_threshold: 0.45
        """)
        with pytest.raises(PolicyError, match="fine_threshold"):
            load_policy(p)

    def test_threshold_out_of_range_raises(self, tmp_path):
        p = write_yaml(tmp_path, """
            routing:
              fine_threshold: 1.5
        """)
        with pytest.raises(PolicyError, match="fine_threshold"):
            load_policy(p)

    def test_missing_routing_key_raises(self, tmp_path):
        p = write_yaml(tmp_path, "quality: {}")
        with pytest.raises(PolicyError, match="routing"):
            load_policy(p)

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_policy(tmp_path / "nonexistent.yaml")

    def test_invalid_yaml_raises(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("routing: [this: is: not: valid")
        with pytest.raises(PolicyError):
            load_policy(p)


# ── Round-trip: save RoutingConfig → load as YAML ────────────────────────────

class TestRoundTrip:
    def test_config_to_yaml_and_back(self, tmp_path):
        from code_capsules.controller.yaml_dsl import config_to_yaml
        cfg = RoutingConfig(
            fine_threshold=0.18,
            compound_threshold=0.50,
            bash_sequential_threshold=0.35,
            compound_min_distinct_files=3,
            quality_lint_weight=0.15,
        )
        yaml_path = tmp_path / "out.yaml"
        config_to_yaml(cfg, yaml_path)
        cfg2 = load_policy(yaml_path)
        assert cfg2.fine_threshold == 0.18
        assert cfg2.compound_threshold == 0.50
        assert cfg2.bash_sequential_threshold == 0.35
        assert cfg2.compound_min_distinct_files == 3
        assert cfg2.quality_lint_weight == 0.15

    def test_yaml_output_is_readable(self, tmp_path):
        from code_capsules.controller.yaml_dsl import config_to_yaml
        yaml_path = tmp_path / "out.yaml"
        config_to_yaml(DEFAULT_CONFIG, yaml_path)
        text = yaml_path.read_text()
        assert "routing:" in text
        assert "fine_threshold:" in text
        assert "compound_gate:" in text


# ── Workload routing block (classifier / routing_strategy / routes / default) ──

class TestRoutingBlock:
    def test_parses_classifier_strategy_routes_default(self, tmp_path):
        p = write_yaml(tmp_path, """
            routing:
              classifier: yaml_mapping
              routing_strategy: rule_based
              routes:
                - when: scientific_workload
                  use: stuck_signal_injection
              default: two_pass_critique
        """)
        cfg = load_policy(p)
        assert cfg.classifier == "yaml_mapping"
        assert cfg.routing_strategy == "rule_based"
        assert cfg.routes == [{"when": "scientific_workload", "use": "stuck_signal_injection"}]
        assert cfg.route_default == "two_pass_critique"

    def test_defaults_when_routing_keys_absent(self, tmp_path):
        cfg = load_policy(write_yaml(tmp_path, "routing: {}"))
        assert cfg.classifier is None
        assert cfg.routing_strategy == "rule_based"
        assert cfg.routes == []
        assert cfg.route_default is None

    def test_unregistered_classifier_raises(self, tmp_path):
        p = write_yaml(tmp_path, """
            routing:
              classifier: no_such_classifier
        """)
        with pytest.raises(PolicyError, match="classifier"):
            load_policy(p)

    def test_unregistered_routing_strategy_raises(self, tmp_path):
        p = write_yaml(tmp_path, """
            routing:
              routing_strategy: no_such_strategy
        """)
        with pytest.raises(PolicyError, match="routing_strategy"):
            load_policy(p)

    def test_route_use_must_be_a_registered_variant(self, tmp_path):
        p = write_yaml(tmp_path, """
            routing:
              routes:
                - when: scientific_workload
                  use: not_a_variant
        """)
        with pytest.raises(PolicyError, match="not_a_variant"):
            load_policy(p)

    def test_malformed_routes_raise(self, tmp_path):
        p = write_yaml(tmp_path, """
            routing:
              routes:
                - just_a_string
        """)
        with pytest.raises(PolicyError, match="routes"):
            load_policy(p)
