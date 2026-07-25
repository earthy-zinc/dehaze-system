package com.pei.dehaze.sdk.utils;

import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;

public class TokenManagerTest {

    @Before
    public void setUp() {
        TokenManager.clearAll();
        TokenManager.setSessionInvalidListener(null);
    }

    @Test
    public void testSessionOperations() {
        String testSessionId = "test_session_id_value";

        assertFalse(TokenManager.hasToken());
        assertNull(TokenManager.getSessionId());

        TokenManager.setSessionId(testSessionId);
        assertTrue(TokenManager.hasToken());
        assertEquals(testSessionId, TokenManager.getSessionId());

        TokenManager.clearAll();
        assertFalse(TokenManager.hasToken());
        assertNull(TokenManager.getSessionId());

        TokenManager.setSessionId("");
        assertFalse(TokenManager.hasToken());
    }

    @Test
    public void testTriggerSessionInvalidClearsSession() {
        TokenManager.setSessionId("session_to_invalidate");
        assertTrue(TokenManager.hasToken());

        TokenManager.triggerSessionInvalid();

        assertFalse(TokenManager.hasToken());
        assertNull(TokenManager.getSessionId());
    }
}
