package com.pei.dehaze.ui.login;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.atLeastOnce;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.verify;

import androidx.arch.core.executor.testing.InstantTaskExecutorRule;
import androidx.lifecycle.Observer;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.api.AuthAPI;
import com.pei.dehaze.sdk.network.ApiException;
import com.pei.dehaze.sdk.model.auth.CaptchaResponse;
import com.pei.dehaze.sdk.model.auth.LoginRequest;
import com.pei.dehaze.sdk.model.auth.LoginResponse;

import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.MockedStatic;
import org.mockito.junit.MockitoJUnitRunner;

@RunWith(MockitoJUnitRunner.class)
public class LoginViewModelIntegrationTest {

    // 确保 LiveData 在测试中立即执行
    @Rule
    public InstantTaskExecutorRule instantExecutorRule = new InstantTaskExecutorRule();

    private LoginViewModel loginViewModel;

    @Mock
    private Observer<Boolean> loadingObserver;

    @Mock
    private Observer<String> errorObserver;

    @Mock
    private Observer<Boolean> successObserver;

    @Mock
    private Observer<String> captchaImageObserver;

    @Before
    public void setUp() {
        loginViewModel = new LoginViewModel();
    }

    @Test
    public void testLoadCaptchaSuccess() {
        // 模拟 AuthAPI.getCaptcha 成功响应
        CaptchaResponse mockResponse = new CaptchaResponse();
        mockResponse.setCaptchaKey("test_key");
        mockResponse.setCaptchaBase64("test_base64_image");

        try (MockedStatic<AuthAPI> mockedAuthAPI = mockStatic(AuthAPI.class)) {
            mockedAuthAPI.when(() -> AuthAPI.getCaptcha(any()))
                    .thenAnswer(invocation -> {
                        ApiCallback<CaptchaResponse> callback = invocation.getArgument(0);
                        callback.onSuccess(mockResponse);
                        return null;
                    });

            // 执行加载验证码
            loginViewModel.loadCaptcha();

            // 验证结果
            assertEquals("test_key", loginViewModel.getCaptchaKey().getValue());
            assertEquals("test_base64_image", loginViewModel.getCaptchaImage().getValue());
        }
    }

    @Test
    public void testLoadCaptchaError() {
        try (MockedStatic<AuthAPI> mockedAuthAPI = mockStatic(AuthAPI.class)) {
            mockedAuthAPI.when(() -> AuthAPI.getCaptcha(any()))
                    .thenAnswer(invocation -> {
                        ApiCallback<CaptchaResponse> callback = invocation.getArgument(0);
                        callback.onError("A0200", "Bad Request");
                        return null;
                    });

            // 观察错误状态
            loginViewModel.getLoginError().observeForever(errorObserver);

            // 执行加载验证码
            loginViewModel.loadCaptcha();

            // 验证错误回调被调用（经过 RepositoryAdapters.wrap + ErrorUtils 解析，A0200 映射为 "用户登录异常"）
            verify(errorObserver).onChanged("获取验证码失败: 用户登录异常");
        }
    }

    @Test
    public void testLoginSuccess() {
        // 设置表单数据
        loginViewModel.getUsername().setValue("admin");
        loginViewModel.getPassword().setValue("123456");
        loginViewModel.getCaptchaCode().setValue("abcd");

        // 模拟 AuthAPI.login 成功响应
        LoginResponse mockResponse = new LoginResponse();
        mockResponse.setSessionId("test_session_id");

        try (MockedStatic<AuthAPI> mockedAuthAPI = mockStatic(AuthAPI.class)) {
            mockedAuthAPI.when(() -> AuthAPI.login(any(LoginRequest.class), any()))
                    .thenAnswer(invocation -> {
                        LoginRequest request = invocation.getArgument(0);
                        ApiCallback<LoginResponse> callback = invocation.getArgument(1);
                        assertEquals("admin", request.getUsername());
                        assertEquals("123456", request.getPassword());
                        assertEquals("abcd", request.getCaptchaCode());
                        callback.onSuccess(mockResponse);
                        return null;
                    });

            // 观察状态变化
            loginViewModel.getLoading().observeForever(loadingObserver);
            loginViewModel.getLoginSuccess().observeForever(successObserver);

            // 执行登录
            loginViewModel.login();

            // 验证加载状态变化
            verify(loadingObserver, atLeastOnce()).onChanged(true);
            verify(loadingObserver, atLeastOnce()).onChanged(false);
            verify(successObserver, atLeastOnce()).onChanged(true);
        }
    }

    @Test
    public void testLoginError() {
        // 设置表单数据
        loginViewModel.getUsername().setValue("admin");
        loginViewModel.getPassword().setValue("123456");
        loginViewModel.getCaptchaCode().setValue("abcd");

        // 模拟 AuthAPI.login 错误响应
        try (MockedStatic<AuthAPI> mockedAuthAPI = mockStatic(AuthAPI.class)) {
            mockedAuthAPI.when(() -> AuthAPI.login(any(LoginRequest.class), any()))
                    .thenAnswer(invocation -> {
                        ApiCallback<LoginResponse> callback = invocation.getArgument(1);
                        callback.onError("A0201", "Unauthorized");
                        return null;
                    });

            // 观察状态变化
            loginViewModel.getLoading().observeForever(loadingObserver);
            loginViewModel.getLoginError().observeForever(errorObserver);

            // 执行登录
            loginViewModel.login();

            // 验证状态变化
            verify(loadingObserver, atLeastOnce()).onChanged(true);
            verify(loadingObserver, atLeastOnce()).onChanged(false);
            verify(errorObserver, atLeastOnce()).onChanged("登录失败: Unauthorized");
        }
    }

    @Test
    public void testLoginNetworkFailure() {
        // 设置表单数据
        loginViewModel.getUsername().setValue("admin");
        loginViewModel.getPassword().setValue("123456");
        loginViewModel.getCaptchaCode().setValue("abcd");

        // 模拟网络错误
        try (MockedStatic<AuthAPI> mockedAuthAPI = mockStatic(AuthAPI.class)) {
            mockedAuthAPI.when(() -> AuthAPI.login(any(LoginRequest.class), any()))
                    .thenAnswer(invocation -> {
                        ApiCallback<LoginResponse> callback = invocation.getArgument(1);
                        callback.onFailure(new ApiException(0, "Network error"));
                        return null;
                    });

            // 观察状态变化
            loginViewModel.getLoading().observeForever(loadingObserver);
            loginViewModel.getLoginError().observeForever(errorObserver);

            // 执行登录
            loginViewModel.login();

            // 验证状态变化（RepositoryCallback 统一为 onError，前缀为 "登录失败"；ErrorUtils 将 httpCode=0 映射为 "网络连接失败，请检查网络设置"）
            verify(loadingObserver, atLeastOnce()).onChanged(true);
            verify(loadingObserver, atLeastOnce()).onChanged(false);
            verify(errorObserver, atLeastOnce()).onChanged("登录失败: 网络连接失败，请检查网络设置");
        }
    }

    @Test
    public void testFormValidation() {
        // 测试表单验证失败的情况
        loginViewModel.getUsername().setValue("");  // 无效用户名
        loginViewModel.getPassword().setValue("123");  // 密码太短
        loginViewModel.getCaptchaCode().setValue("");  // 验证码为空

        // 观察错误状态
        loginViewModel.getLoginError().observeForever(errorObserver);

        // 执行登录
        loginViewModel.login();

        // 验证错误信息
        verify(errorObserver).onChanged("用户名格式不正确");
    }
}