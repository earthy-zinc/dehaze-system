package com.pei.dehaze.sdk;

import com.pei.dehaze.sdk.utils.TokenManager;

import org.junit.Before;
import org.junit.Test;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

import static org.junit.Assert.*;
import static org.mockito.Mockito.*;

public class DehazeSDKTest {

    @Mock
    private DehazeSDK dehazeSDK;

    @Before
    public void setUp() {
        MockitoAnnotations.initMocks(this);
        TokenManager.clearAll();
        TokenManager.setSessionInvalidListener(null);
    }

    @Test
    public void testSDKInitialization() {
        try {
            DehazeSDK.initialize(new DehazeSDK.Builder().setBaseUrl("http://127.0.0.1:8989"));
            assertNotNull(DehazeSDK.getInstance());
        } catch (Exception e) {
            fail("SDK初始化失败: " + e.getMessage());
        }
    }

    @Test
    public void testTokenManager() {
        String testSessionId = "test_session_id_12345";

        TokenManager.setSessionId(testSessionId);
        assertTrue(TokenManager.hasToken());
        assertEquals(testSessionId, TokenManager.getSessionId());

        TokenManager.clearAll();
        assertFalse(TokenManager.hasToken());
        assertNull(TokenManager.getSessionId());
    }
}
