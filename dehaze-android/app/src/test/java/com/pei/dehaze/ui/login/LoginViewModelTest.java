package com.pei.dehaze.ui.login;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;
import org.robolectric.annotation.Config;

@RunWith(RobolectricTestRunner.class)
@Config(sdk = 28)
public class LoginViewModelTest {

    private LoginViewModel loginViewModel;

    @Before
    public void setUp() {
        loginViewModel = new LoginViewModel();
    }

    @Test
    public void testValidUsername() {
        // 测试有效的用户名
        loginViewModel.setUsername("admin");
        assertEquals("admin", loginViewModel.getUsername().getValue());
    }

    @Test
    public void testValidPassword() {
        // 测试有效的密码
        loginViewModel.setPassword("123456");
        assertEquals("123456", loginViewModel.getPassword().getValue());
    }

    @Test
    public void testValidCaptchaCode() {
        // 测试有效的验证码
        loginViewModel.setCaptchaCode("abcd");
        assertEquals("abcd", loginViewModel.getCaptchaCode().getValue());
    }

    @Test
    public void testFormValidationWithValidData() {
        // 测试表单验证 - 有效数据
        loginViewModel.setUsername("admin");
        loginViewModel.setPassword("123456");
        loginViewModel.setCaptchaCode("abcd");
        
        // 注意：实际的验证逻辑在 login() 方法中执行
        // 这里我们只是测试数据是否正确设置
        assertEquals("admin", loginViewModel.getUsername().getValue());
        assertEquals("123456", loginViewModel.getPassword().getValue());
        assertEquals("abcd", loginViewModel.getCaptchaCode().getValue());
    }

    @Test
    public void testFormValidationWithInvalidUsername() {
        // 测试表单验证 - 无效用户名（空）
        loginViewModel.setUsername("");
        assertEquals("", loginViewModel.getUsername().getValue());
    }

    @Test
    public void testFormValidationWithInvalidPassword() {
        // 测试表单验证 - 无效密码（太短）
        loginViewModel.setPassword("123");
        assertEquals("123", loginViewModel.getPassword().getValue());
    }

    @Test
    public void testFormValidationWithEmptyCaptcha() {
        // 测试表单验证 - 空验证码
        loginViewModel.setCaptchaCode("");
        assertEquals("", loginViewModel.getCaptchaCode().getValue());
    }

    @Test
    public void testInitialState() {
        // 测试初始状态
        assertEquals("", loginViewModel.getUsername().getValue());
        assertEquals("", loginViewModel.getPassword().getValue());
        assertEquals("", loginViewModel.getCaptchaCode().getValue());
        assertEquals("", loginViewModel.getCaptchaKey().getValue());
        assertEquals("", loginViewModel.getCaptchaImage().getValue());
        assertFalse(loginViewModel.getLoading().getValue());
        assertEquals("", loginViewModel.getLoginError().getValue());
        assertFalse(loginViewModel.getLoginSuccess().getValue());
    }
}