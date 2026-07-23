package com.pei.dehaze.sdk.utils;

import com.pei.dehaze.sdk.network.ApiException;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 * 工具类测试
 */
public class ErrorUtilsTest {
    
    @Test
    public void testParseError() {
        // 测试解析网络错误
        ApiException exception = new ApiException(404, "Not Found");
        String message = ErrorUtils.parseError(exception);
        assertNotNull("错误消息不应为null", message);
        assertFalse("错误消息不应为空", message.isEmpty());
        
        // 测试null异常
        String nullMessage = ErrorUtils.parseError(null);
        assertEquals("应该返回默认错误消息", "未知错误", nullMessage);
        
        // 测试不同错误码
        ApiException unauthorized = new ApiException(401, "Unauthorized");
        String unauthorizedMessage = ErrorUtils.parseError(unauthorized);
        assertEquals("应该返回未授权消息", "未授权访问，请重新登录", unauthorizedMessage);
        
        ApiException networkError = new ApiException(-1, "Network error");
        String networkMessage = ErrorUtils.parseError(networkError);
        assertEquals("应该返回网络错误消息", "网络连接失败，请检查网络设置", networkMessage);
    }
}