import pytest

from phoibe.layered.application.factory import RuleRegistry
from phoibe.layered.rules.rule import ValidationRule


class TestRuleRegistryEdgeCases:

    def setup_method(self):
        RuleRegistry.clear()

    def teardown_method(self):
        RuleRegistry.clear()

    def test_duplicate_registration_overwrites_silently(self):

        @RuleRegistry.register("duplicate")
        class FirstRule(ValidationRule):
            @property
            def name(self):
                return "first"

            def execute(self, df, context):
                pass

        @RuleRegistry.register("duplicate")
        class SecondRule(ValidationRule):
            @property
            def name(self):
                return "second"

            def execute(self, df, context):
                pass

        rule_class = RuleRegistry.get("duplicate")
        assert rule_class.__name__ == "SecondRule"

    def test_registration_with_special_characters_in_name(self):

        @RuleRegistry.register("rule-with-dashes")
        class RuleWithDashes(ValidationRule):
            @property
            def name(self):
                return "rule-with-dashes"

            def execute(self, df, context):
                pass

        @RuleRegistry.register("rule_with_underscores")
        class RuleWithUnderscores(ValidationRule):
            @property
            def name(self):
                return "rule_with_underscores"

            def execute(self, df, context):
                pass

        @RuleRegistry.register("rule.with.dots")
        class RuleWithDots(ValidationRule):
            @property
            def name(self):
                return "rule.with.dots"

            def execute(self, df, context):
                pass

        assert "rule-with-dashes" in RuleRegistry.list_rules()
        assert "rule_with_underscores" in RuleRegistry.list_rules()
        assert "rule.with.dots" in RuleRegistry.list_rules()

    def test_registration_with_numeric_name(self):

        @RuleRegistry.register("rule123")
        class NumericRule(ValidationRule):
            @property
            def name(self):
                return "rule123"

            def execute(self, df, context):
                pass

        assert "rule123" in RuleRegistry.list_rules()

    def test_registration_with_very_long_name(self):
        long_name = "this_is_a_very_long_rule_name_" * 5

        @RuleRegistry.register(long_name)
        class LongNameRule(ValidationRule):
            @property
            def name(self):
                return long_name

            def execute(self, df, context):
                pass

        assert long_name in RuleRegistry.list_rules()

    def test_get_unknown_rule_error_message_helpful(self):

        @RuleRegistry.register("available1")
        class Rule1(ValidationRule):
            name = "available1"

            def execute(self, df, context):
                pass

        @RuleRegistry.register("available2")
        class Rule2(ValidationRule):
            name = "available2"

            def execute(self, df, context):
                pass

        with pytest.raises(KeyError) as exc_info:
            RuleRegistry.get("typo_rule")

        error_msg = str(exc_info.value)
        assert "typo_rule" in error_msg
        assert "available1" in error_msg
        assert "available2" in error_msg

    def test_get_unknown_rule_when_registry_empty(self):
        with pytest.raises(KeyError) as exc_info:
            RuleRegistry.get("any_rule")

        error_msg = str(exc_info.value)
        assert "any_rule" in error_msg

    def test_clear_idempotent(self):
        RuleRegistry.clear()
        RuleRegistry.clear()
        RuleRegistry.clear()

        assert RuleRegistry.list_rules() == []

    def test_clear_and_re_register(self):
        @RuleRegistry.register("rule1")
        class Rule1(ValidationRule):
            name = "rule1"

            def execute(self, df, context):
                pass

        assert "rule1" in RuleRegistry.list_rules()
        RuleRegistry.clear()
        assert "rule1" not in RuleRegistry.list_rules()

        @RuleRegistry.register("rule1")
        class Rule1Again(ValidationRule):
            name = "rule1"

            def execute(self, df, context):
                pass

        assert "rule1" in RuleRegistry.list_rules()

    def test_list_rules_returns_new_list_each_time(self):
        @RuleRegistry.register("rule1")
        class Rule1(ValidationRule):
            name = "rule1"

            def execute(self, df, context):
                pass

        list1 = RuleRegistry.list_rules()
        list2 = RuleRegistry.list_rules()

        assert list1 == list2
        assert list1 is not list2

    def test_modifying_list_does_not_affect_registry(self):
        @RuleRegistry.register("rule1")
        class Rule1(ValidationRule):
            name = "rule1"

            def execute(self, df, context):
                pass

        rules_list = RuleRegistry.list_rules()
        rules_list.append("fake_rule")
        rules_list.clear()

        assert RuleRegistry.list_rules() == ["rule1"]

    def test_rule_names_are_case_sensitive(self):
        @RuleRegistry.register("MyRule")
        class UpperRule(ValidationRule):
            name = "MyRule"

            def execute(self, df, context):
                pass

        @RuleRegistry.register("myrule")
        class LowerRule(ValidationRule):
            name = "myrule"

            def execute(self, df, context):
                pass

        assert "MyRule" in RuleRegistry.list_rules()
        assert "myrule" in RuleRegistry.list_rules()
        assert RuleRegistry.get("MyRule") is UpperRule
        assert RuleRegistry.get("myrule") is LowerRule

    def test_can_register_empty_string_name(self):

        @RuleRegistry.register("")
        class EmptyNameRule(ValidationRule):
            name = ""

            def execute(self, df, context):
                pass

        assert "" in RuleRegistry.list_rules()
        assert RuleRegistry.get("") is EmptyNameRule
