"""Unit tests for api/registry.py (the name -> implementation registry)."""
import pytest

from code_capsules.api import registry as R


class TestRegistryMechanism:
    def test_register_and_get_roundtrip(self):
        class _Stub:
            name = "test_stub_classifier"
            def classify(self, task):  # pragma: no cover - not invoked
                return ""
        stub = _Stub()
        R.register("classifier", stub)
        assert R.get("classifier", "test_stub_classifier") is stub

    def test_get_unknown_name_raises_keyerror(self):
        with pytest.raises(KeyError):
            R.get("quality_gate", "definitely_not_registered")

    def test_unknown_kind_raises_valueerror(self):
        with pytest.raises(ValueError):
            R.get("not_a_kind", "x")

    def test_register_without_name_raises(self):
        with pytest.raises(ValueError):
            R.register("classifier", object())

    def test_list_registered_is_sorted(self):
        names = R.list_registered("cost_model")
        assert names == sorted(names)


class TestBuiltinAliases:
    def test_diverse_agreement_and_alias_resolve_same_instance(self):
        canonical = R.get("cascade_trigger", "diverse_agreement")
        alias = R.get("cascade_trigger", "cross_sample_agreement")
        assert canonical is alias
        assert canonical.name == "diverse_agreement"

    def test_lazy_builtins_registered_on_first_access(self):
        # All eight registries populate on first access (no explicit init needed).
        for kind in ("classifier", "routing_strategy", "variant", "signal",
                     "quality_gate", "cascade_trigger", "cost_model", "model_client"):
            assert R.list_registered(kind), f"{kind} registry is empty"
