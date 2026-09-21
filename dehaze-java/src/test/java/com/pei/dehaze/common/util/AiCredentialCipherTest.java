package com.pei.dehaze.common.util;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 凭据加解密跨端互认测试。
 *
 * <p>固定密文向量由 python 端 {@code app/infrastructure/crypto/aes_cipher.py}（同密钥派生、
 * AES-256-CBC + PKCS7 + base64(iv||ct)）生成，用于锁定 java 与 python 的密文格式一致：
 * 任一端写入的密文，另一端都必须能解开（供应商 Key / MCP 凭据均存该密文）。
 */
class AiCredentialCipherTest {

    private static final String KEY = "dehaze-cross-end-test-key";

    /** python 以固定 IV 生成的密文（iv=00..0f） */
    private static final String PYTHON_CIPHER_KEY = "AAECAwQFBgcICQoLDA0ODz1AYgtqdeBmh/6NKWzlW3NCUUnNiiaerx4P+CPjrzaD";
    private static final String PYTHON_PLAIN_KEY = "sk-test-cross-end-key-1234";

    /** python 生成的 MCP 凭据 JSON 密文（iv=10..1f） */
    private static final String PYTHON_CIPHER_JSON = "EBESExQVFhcYGRobHB0eH1nrROjtznJgrr73E0wIRlIucsxBTNUEx4LrNjGw2y4+";
    private static final String PYTHON_PLAIN_JSON = "{\"api_key\":\"secret\"}";

    private final AiCredentialCipher cipher = new AiCredentialCipher(KEY);

    @Test
    void shouldDecryptPythonCiphertext() {
        assertEquals(PYTHON_PLAIN_KEY, cipher.decrypt(PYTHON_CIPHER_KEY));
        assertEquals(PYTHON_PLAIN_JSON, cipher.decrypt(PYTHON_CIPHER_JSON));
    }

    @Test
    void shouldRoundTripOwnCiphertext() {
        String cipherText = cipher.encrypt(PYTHON_PLAIN_KEY);
        assertNotEquals(PYTHON_PLAIN_KEY, cipherText);
        assertEquals(PYTHON_PLAIN_KEY, cipher.decrypt(cipherText));
    }

    @Test
    void shouldUseRandomIvPerEncryption() {
        assertNotEquals(cipher.encrypt(PYTHON_PLAIN_KEY), cipher.encrypt(PYTHON_PLAIN_KEY));
    }

    @Test
    void shouldHashAndMaskLikePython() {
        assertEquals("856515f37e98ccb7c9b21fdeaf1960856a9216fe82dcaac672849580b1e0f329",
                cipher.hashKey(PYTHON_PLAIN_KEY));
        assertEquals("sk-test-...", cipher.maskKey(PYTHON_PLAIN_KEY));
        assertTrue(cipher.maskKey("short").startsWith("short"));
    }
}
