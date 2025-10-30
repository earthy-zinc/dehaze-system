import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.service.auth_service import AuthService
from flask import Flask


class TestAuthService(unittest.TestCase):

    def setUp(self):
        """测试前准备"""
        self.username = "testuser"
        self.password = "testpass123"
        self.user_id = 1

    def test_login_success(self):
        """测试登录成功"""
        # 创建应用上下文
        app = Flask(__name__)
        app.config['SECRET_KEY'] = 'test_secret'

        # 手动模拟对象
        with patch('app.service.auth_service.SysUser') as mock_sys_user, \
                patch('app.service.auth_service.jwt') as mock_jwt:
            # 准备模拟数据
            mock_user = MagicMock()
            mock_user.id = self.user_id
            mock_user.status = 1
            mock_user.check_password.return_value = True
            mock_sys_user.query.filter_by.return_value.first.return_value = mock_user
            mock_jwt.encode.return_value = "fake_token"

            # 在应用上下文中调用被测试方法
            with app.app_context():
                result = AuthService.login(self.username, self.password)

            # 验证结果
            self.assertEqual(result['tokenType'], 'Bearer')
            self.assertEqual(result['accessToken'], 'fake_token')
            mock_sys_user.query.filter_by.assert_called_with(username=self.username, deleted=0)
            mock_user.check_password.assert_called_with(self.password)

    @patch('app.service.auth_service.SysUser')
    def test_login_invalid_user(self, mock_sys_user):
        """测试用户不存在"""
        # 准备模拟数据
        mock_sys_user.query.filter_by.return_value.first.return_value = None

        # 验证异常
        with self.assertRaises(Exception) as context:
            AuthService.login(self.username, self.password)

        self.assertIn("用户名或密码错误", str(context.exception))

    @patch('app.service.auth_service.SysUser')
    def test_login_invalid_password(self, mock_sys_user):
        """测试密码错误"""
        # 准备模拟数据
        mock_user = MagicMock()
        mock_user.check_password.return_value = False
        mock_sys_user.query.filter_by.return_value.first.return_value = mock_user

        # 验证异常
        with self.assertRaises(Exception) as context:
            AuthService.login(self.username, self.password)

        self.assertIn("用户名或密码错误", str(context.exception))

    @patch('app.service.auth_service.SysUser')
    def test_login_disabled_user(self, mock_sys_user):
        """测试用户被禁用"""
        # 准备模拟数据
        mock_user = MagicMock()
        mock_user.status = 0  # 用户被禁用
        mock_user.check_password.return_value = True
        mock_sys_user.query.filter_by.return_value.first.return_value = mock_user

        # 验证异常
        with self.assertRaises(Exception) as context:
            AuthService.login(self.username, self.password)

        self.assertIn("用户已被禁用", str(context.exception))

    @patch('app.service.auth_service.get_current_user_id')
    def test_logout_success(self, mock_get_current_user_id):
        """测试注销成功"""
        # 准备模拟数据
        mock_get_current_user_id.return_value = self.user_id

        # 创建应用上下文
        app = Flask(__name__)

        # 在应用上下文中调用被测试方法（不会抛出异常即为成功）
        with app.app_context():
            try:
                AuthService.logout()
            except Exception:
                self.fail("logout() raised Exception unexpectedly!")

    @patch('app.service.auth_service.get_current_user_id')
    def test_logout_no_session(self, mock_get_current_user_id):
        """测试无有效会话时注销"""
        # 准备模拟数据
        mock_get_current_user_id.return_value = None

        # 创建应用上下文
        app = Flask(__name__)

        # 在应用上下文中调用被测试方法
        with app.app_context():
            try:
                AuthService.logout()
            except Exception as e:
                # 根据实现，如果用户ID不存在会抛出异常，这在测试中是可以接受的
                self.assertIn("未找到有效的用户会话", str(e))

    def test_get_captcha(self):
        """测试获取验证码"""
        # 创建应用上下文
        app = Flask(__name__)
        mock_redis = MagicMock()
        app.extensions = {'redis_client': mock_redis}

        # 在应用上下文中调用被测试方法
        with app.app_context():
            result = AuthService.get_captcha()

        # 验证结果
        self.assertIn('captchaKey', result)
        self.assertIn('captchaBase64', result)
        self.assertTrue(result['captchaBase64'].startswith('data:image/jpeg;base64,'))
        mock_redis.setex.assert_called_once()

    def test_verify_captcha_success(self):
        """测试验证码验证成功"""
        # 准备模拟数据
        app = Flask(__name__)
        mock_redis = MagicMock()
        app.extensions = {'redis_client': mock_redis}
        captcha_key = "test_key"
        captcha_code = "ABCD"
        mock_redis.get.return_value = b'ABCD'

        # 在应用上下文中调用被测试方法
        with app.app_context():
            result = AuthService.verify_captcha(captcha_key, captcha_code)

        # 验证结果
        self.assertTrue(result)
        mock_redis.get.assert_called_with(f"captcha:{captcha_key}")
        mock_redis.delete.assert_called_with(f"captcha:{captcha_key}")

    def test_verify_captcha_failure(self):
        """测试验证码验证失败"""
        # 准备模拟数据
        app = Flask(__name__)
        mock_redis = MagicMock()
        app.extensions = {'redis_client': mock_redis}
        captcha_key = "test_key"
        captcha_code = "ABCD"
        mock_redis.get.return_value = b'1234'  # 不匹配的验证码

        # 在应用上下文中调用被测试方法
        with app.app_context():
            result = AuthService.verify_captcha(captcha_key, captcha_code)

        # 验证结果
        self.assertFalse(result)
        mock_redis.get.assert_called_with(f"captcha:{captcha_key}")

    def test_verify_captcha_expired(self):
        """测试验证码过期"""
        # 准备模拟数据
        app = Flask(__name__)
        mock_redis = MagicMock()
        app.extensions = {'redis_client': mock_redis}
        captcha_key = "test_key"
        captcha_code = "ABCD"
        mock_redis.get.return_value = None  # 验证码不存在（过期或不存在）

        # 在应用上下文中调用被测试方法
        with app.app_context():
            result = AuthService.verify_captcha(captcha_key, captcha_code)

        # 验证结果
        self.assertFalse(result)
        mock_redis.get.assert_called_with(f"captcha:{captcha_key}")


if __name__ == '__main__':
    unittest.main()
