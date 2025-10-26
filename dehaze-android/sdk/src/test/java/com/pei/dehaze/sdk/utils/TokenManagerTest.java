package com.pei.dehaze.sdk.utils;

import org.junit.Test;
import static org.junit.Assert.*;

/**
 * Token管理器测试
 */
public class TokenManagerTest {
    
    @Test
    public void testTokenOperations() {
        // 测试Token操作
        String testToken = "test_token_value";
        
        // 初始状态
        assertFalse("初始时Token不应存在", TokenManager.hasToken());
        assertNull("初始时Token应为null", TokenManager.getToken());
        
        // 设置Token
        TokenManager.setToken(testToken);
        assertTrue("设置后Token应存在", TokenManager.hasToken());
        assertEquals("Token值应匹配", testToken, TokenManager.getToken());
        
        // 清除Token
        TokenManager.clearToken();
        assertFalse("清除后Token不应存在", TokenManager.hasToken());
        assertNull("清除后Token应为null", TokenManager.getToken());
        
        // 测试空字符串Token
        TokenManager.setToken("");
        assertFalse("空字符串Token应视为不存在", TokenManager.hasToken());
    }
}