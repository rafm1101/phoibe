import pytest

from phoibe.layered.application.registry import RuleRegistry
from phoibe.layered.rules.rule import ValidationRule


class TestRuleRegistry:

    def setup_method(self):
        RuleRegistry.clear()

    def teardown_method(self):
        RuleRegistry.clear()

    def test_register_decorator_registers_rule(self):
        @RuleRegistry.register("test_rule")
        class TestRule(ValidationRule):
            @property
            def name(self):
                return "test_rule"

            def execute(self, df, context):
                pass

        assert "test_rule" in RuleRegistry.list_rules()

    def test_register_returns_class_unchanged(self):
        @RuleRegistry.register("test_rule")
        class TestRule(ValidationRule):
            @property
            def name(self):
                return "test_rule"

            def execute(self, df, context):
                pass

        instance = TestRule(points=13)
        assert instance.name == "test_rule"

    def test_register_multiple_rules(self):
        @RuleRegistry.register("rule1")
        class Rule1(ValidationRule):
            @property
            def name(self):
                return "rule1"

            def execute(self, df, context):
                pass

        @RuleRegistry.register("rule2")
        class Rule2(ValidationRule):
            @property
            def name(self):
                return "rule2"

            def execute(self, df, context):
                pass

        rules = RuleRegistry.list_rules()
        assert "rule1" in rules
        assert "rule2" in rules
        assert len(rules) == 2

    def test_register_overwrites_duplicate_name(self):

        @RuleRegistry.register("duplicate")
        class Rule1(ValidationRule):
            @property
            def name(self):
                return "duplicate"

            def execute(self, df, context):
                return "rule1"

        @RuleRegistry.register("duplicate")
        class Rule2(ValidationRule):
            @property
            def name(self):
                return "duplicate"

            def execute(self, df, context):
                return "rule2"

        rule_class = RuleRegistry.get("duplicate")
        assert rule_class is Rule2

    def test_get_returns_rule_class(self):
        @RuleRegistry.register("test_rule")
        class TestRule(ValidationRule):
            @property
            def name(self):
                return "test_rule"

            def execute(self, df, context):
                pass

        rule_class = RuleRegistry.get("test_rule")
        assert rule_class is TestRule

    def test_get_raises_given_unknown_rule(self):
        with pytest.raises(KeyError, match="not found in registry"):
            RuleRegistry.get("nonexistent_rule")

    def test_get_error_message_lists_available_rules(self):
        @RuleRegistry.register("available_rule")
        class TestRule(ValidationRule):
            @property
            def name(self):
                return "available_rule"

            def execute(self, df, context):
                pass

        with pytest.raises(KeyError, match="available_rule"):
            RuleRegistry.get("missing_rule")

    def test_list_rules_returns_sorted_list(self):
        @RuleRegistry.register("zebra")
        class Rule1(ValidationRule):
            name = "zebra"

            def execute(self, df, context):
                pass

        @RuleRegistry.register("alpha")
        class Rule2(ValidationRule):
            name = "alpha"

            def execute(self, df, context):
                pass

        @RuleRegistry.register("beta")
        class Rule3(ValidationRule):
            name = "beta"

            def execute(self, df, context):
                pass

        rules = RuleRegistry.list_rules()

        assert rules == ["alpha", "beta", "zebra"]

    def test_list_rules_empty_when_no_rules(self):
        rules = RuleRegistry.list_rules()
        assert rules == []

    def test_is_registered_returns_true_given_registered_rule(self):
        @RuleRegistry.register("test_rule")
        class TestRule(ValidationRule):
            pass

        assert RuleRegistry.is_registered("test_rule") is True

    def test_is_registered_returns_false_given_unknown_rule(self):
        assert RuleRegistry.is_registered("unknown_rule") is False

    def test_clear_removes_all_rules(self):
        @RuleRegistry.register("rule1")
        class Rule1(ValidationRule):
            name = "rule1"

            def execute(self, df, context):
                pass

        @RuleRegistry.register("rule2")
        class Rule2(ValidationRule):
            name = "rule2"

            def execute(self, df, context):
                pass

        assert len(RuleRegistry.list_rules()) == 2
        RuleRegistry.clear()
        assert RuleRegistry.list_rules() == []
