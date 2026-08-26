from app.service.ai_feedback_service import _build_preference_content
from app.utils.pii import contains_pii, mask_pii


class TestPiiFilter:

    def test_id_card_detected(self):
        assert contains_pii("我的身份证是 11010519491231002X")
        assert "11010519491231002X" not in mask_pii("我的身份证是 11010519491231002X")

    def test_phone_detected(self):
        assert contains_pii("联系电话 13800138000")
        assert "13800138000" not in mask_pii("联系电话 13800138000")

    def test_bank_card_detected(self):
        assert contains_pii("银行卡号 6222021234567890123")
        assert "6222021234567890123" not in mask_pii("银行卡号 6222021234567890123")

    def test_secret_key_detected(self):
        assert contains_pii("我的 API Key 是 sk-abcdefgh12345678")
        assert "sk-abcdefgh12345678" not in mask_pii("我的 API Key 是 sk-abcdefgh12345678")

    def test_password_detected(self):
        assert contains_pii("密码：abc12345")
        masked = mask_pii("密码：abc12345")
        assert "abc12345" not in masked

    def test_normal_text_unaffected(self):
        assert not contains_pii("用户偏好简洁回复")
        assert mask_pii("用户偏好简洁回复") == "用户偏好简洁回复"

    def test_masked_preserves_context(self):
        masked = mask_pii("联系手机 13800138000，欢迎咨询")
        assert "欢迎咨询" in masked
        assert "13800138000" not in masked

    def test_cjk_adjacent_pii_detected(self):
        assert contains_pii("电话13800138000谢谢")
        assert "13800138000" not in mask_pii("电话13800138000谢谢")
        assert "11010519491231002X" not in mask_pii("身份证11010519491231002X号")
        assert "sk-abcdefgh12345678" not in mask_pii("密钥sk-abcdefgh12345678备用")
        assert "6222021234567890123" not in mask_pii("卡号6222021234567890123。")

    def test_no_partial_match_inside_longer_digit_run(self):
        assert mask_pii("编号138001380001结束") == "编号138001380001结束"


class TestFeedbackExtraction:

    def test_too_long_mapping(self):
        assert _build_preference_content(["too_long"], None) == "用户偏好简洁回复"

    def test_with_comment(self):
        content = _build_preference_content(["too_long"], "请精简到3点以内")
        assert "用户偏好简洁回复" in content
        assert "请精简到3点以内" in content

    def test_no_mapped_tag_returns_none(self):
        assert _build_preference_content(["incorrect"], None) is None

    def test_empty_returns_none(self):
        assert _build_preference_content([], None) is None
