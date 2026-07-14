import unittest

from reasoning_agent_template.workflow import (
    _is_identity_question,
)


class IntentRuleTests(unittest.TestCase):
    def test_quoted_chinese_greeting_is_not_identity_question(self):
        self.assertFalse(
            _is_identity_question(
                "请把“你好”换成更友好的表达。"
            )
        )

    def test_quoted_english_greeting_is_not_identity_question(self):
        self.assertFalse(
            _is_identity_question(
                '请把 "hello" 改得更正式一点。'
            )
        )

    def test_standalone_greeting_remains_supported(self):
        self.assertTrue(
            _is_identity_question("你好")
        )
        self.assertTrue(
            _is_identity_question("Hello!")
        )

    def test_real_identity_questions_remain_supported(self):
        self.assertTrue(
            _is_identity_question("你是谁？")
        )
        self.assertTrue(
            _is_identity_question(
                "请介绍一下你自己。"
            )
        )


if __name__ == "__main__":
    unittest.main()
