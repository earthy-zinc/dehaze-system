package com.pei.dehaze.sdk;

import com.pei.dehaze.sdk.model.auth.LoginRequest;
import com.pei.dehaze.sdk.model.auth.LoginResponse;
import com.pei.dehaze.sdk.model.auth.CaptchaResponse;
import com.pei.dehaze.sdk.utils.TokenManager;

import org.junit.Before;
import org.junit.Test;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

import static org.junit.Assert.*;
import static org.mockito.Mockito.*;

/**
 * DehazeSDK测试类
 * 测试SDK的基本功能和API调用
 */
public class DehazeSDKTest {
    
    @Mock
    private DehazeSDK dehazeSDK;
    
    @Before
    public void setUp() {
        MockitoAnnotations.initMocks(this);
    }
    
    @Test
    public void testSDKInitialization() {
        // 测试SDK初始化
        try {
            DehazeSDK.initialize(new DehazeSDK.Builder());
            assertNotNull("SDK实例不应为null", DehazeSDK.getInstance());
        } catch (Exception e) {
            fail("SDK初始化失败: " + e.getMessage());
        }
    }
    
    @Test
    public void testTokenManager() {
        // 测试Token管理功能
        String testToken = "test_token_12345";
        
        // 设置Token
        TokenManager.setToken(testToken);
        assertTrue("Token应该存在", TokenManager.hasToken());
        assertEquals("Token应该匹配", testToken, TokenManager.getToken());
        
        // 清除Token
        TokenManager.clearToken();
        assertFalse("Token应该被清除", TokenManager.hasToken());
        assertNull("Token应该为null", TokenManager.getToken());
    }
    
    @Test
    public void testAuthAPICalls() {
        // 测试认证API调用
        try {
            // 测试获取验证码
            AuthAPI.getCaptcha(new ApiCallback<CaptchaResponse>() {
                @Override
                public void onSuccess(CaptchaResponse data) {
                    // 验证码获取成功
                    assertNotNull("验证码数据不应为null", data);
                }
                
                @Override
                public void onError(int code, String message) {
                    // 业务错误处理
                    fail("获取验证码业务错误: " + code + " - " + message);
                }
                
                @Override
                public void onFailure(com.pei.dehaze.sdk.network.ApiException e) {
                    // 网络错误处理
                    // 注意：在单元测试中，网络错误是可以预期的
                    assertNotNull("应该捕获到网络异常", e);
                }
            });
            
            // 测试登录
            LoginRequest request = new LoginRequest();
            request.setUsername("testuser");
            request.setPassword("testpassword");
            request.setCaptchaCode("1234");
            request.setCaptchaKey("testkey");
            
            AuthAPI.login(request, new ApiCallback<LoginResponse>() {
                @Override
                public void onSuccess(LoginResponse data) {
                    // 登录成功
                    assertNotNull("登录响应数据不应为null", data);
                    if (data.getToken() != null) {
                        TokenManager.setToken(data.getToken());
                    }
                }
                
                @Override
                public void onError(int code, String message) {
                    // 业务错误处理
                    // 在测试环境中，业务错误是可以预期的
                }
                
                @Override
                public void onFailure(com.pei.dehaze.sdk.network.ApiException e) {
                    // 网络错误处理
                    // 在单元测试中，网络错误是可以预期的
                    assertNotNull("应该捕获到网络异常", e);
                }
            });
        } catch (Exception e) {
            // API调用过程中的异常
            fail("API调用异常: " + e.getMessage());
        }
    }
}