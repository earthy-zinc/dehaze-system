package com.pei.dehaze.common.util;

import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import jakarta.annotation.PostConstruct;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import javax.crypto.Cipher;
import javax.crypto.spec.IvParameterSpec;
import javax.crypto.spec.SecretKeySpec;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.SecureRandom;
import java.util.Arrays;
import java.util.Base64;

/**
 * AI 供应商 Key / MCP 凭据的 AES-256-CBC 加解密。
 *
 * <p>与 Python 端 {@code app/infrastructure/crypto/aes_cipher.py} 严格同格式，跨端可互认：
 * 密钥 = SHA256(环境变量 AI_PROVIDER_KEY_ENCRYPTION_KEY)，密文 = base64(IV(16B) || AES-CBC-PKCS7 密文)。
 * 加密密钥仅环境变量注入，绝不落库、不进日志。
 */
@Slf4j
@Component
public class AiCredentialCipher {

    private static final int IV_LENGTH = 16;
    private static final String TRANSFORMATION = "AES/CBC/PKCS5Padding";
    private static final String ALGORITHM = "AES";

    private final String encryptionKey;
    private final SecureRandom secureRandom = new SecureRandom();

    public AiCredentialCipher(@Value("${AI_PROVIDER_KEY_ENCRYPTION_KEY:}") String encryptionKey) {
        this.encryptionKey = encryptionKey;
    }

    @PostConstruct
    public void init() {
        if (encryptionKey == null || encryptionKey.isBlank()) {
            log.warn("AI_PROVIDER_KEY_ENCRYPTION_KEY 未配置，供应商 Key / MCP 凭据加解密将不可用");
        }
    }

    /** AES-256-CBC 加密，返回 base64(iv + ct) */
    public String encrypt(String plaintext) {
        try {
            byte[] iv = new byte[IV_LENGTH];
            secureRandom.nextBytes(iv);
            Cipher cipher = Cipher.getInstance(TRANSFORMATION);
            cipher.init(Cipher.ENCRYPT_MODE, secretKey(), new IvParameterSpec(iv));
            byte[] cipherText = cipher.doFinal(plaintext.getBytes(StandardCharsets.UTF_8));
            byte[] combined = new byte[iv.length + cipherText.length];
            System.arraycopy(iv, 0, combined, 0, iv.length);
            System.arraycopy(cipherText, 0, combined, iv.length, cipherText.length);
            return Base64.getEncoder().encodeToString(combined);
        } catch (Exception e) {
            throw new BusinessException(ResultCode.SYSTEM_EXECUTION_ERROR, "凭据加密失败");
        }
    }

    /** AES-256-CBC 解密（与 Python 端 encrypt 产物互认） */
    public String decrypt(String ciphertext) {
        try {
            byte[] raw = Base64.getDecoder().decode(ciphertext);
            byte[] iv = Arrays.copyOfRange(raw, 0, IV_LENGTH);
            byte[] content = Arrays.copyOfRange(raw, IV_LENGTH, raw.length);
            Cipher cipher = Cipher.getInstance(TRANSFORMATION);
            cipher.init(Cipher.DECRYPT_MODE, secretKey(), new IvParameterSpec(iv));
            return new String(cipher.doFinal(content), StandardCharsets.UTF_8);
        } catch (Exception e) {
            throw new BusinessException(ResultCode.SYSTEM_EXECUTION_ERROR, "凭据解密失败");
        }
    }

    /** SHA256 十六进制哈希（明文查重，不落明文） */
    public String hashKey(String plaintext) {
        try {
            byte[] digest = MessageDigest.getInstance("SHA-256")
                    .digest(plaintext.getBytes(StandardCharsets.UTF_8));
            StringBuilder builder = new StringBuilder(digest.length * 2);
            for (byte b : digest) {
                builder.append(Character.forDigit((b >> 4) & 0xF, 16));
                builder.append(Character.forDigit(b & 0xF, 16));
            }
            return builder.toString();
        } catch (Exception e) {
            throw new BusinessException(ResultCode.SYSTEM_EXECUTION_ERROR, "凭据哈希失败");
        }
    }

    /** 展示用前缀：前 8 位 + "..."（与 Python mask_key 同口径） */
    public String maskKey(String plaintext) {
        return plaintext.length() > 8 ? plaintext.substring(0, 8) + "..." : plaintext + "...";
    }

    private SecretKeySpec secretKey() {
        if (encryptionKey == null || encryptionKey.isBlank()) {
            throw new BusinessException(ResultCode.SYSTEM_EXECUTION_ERROR,
                    "AI_PROVIDER_KEY_ENCRYPTION_KEY 未配置，无法处理凭据");
        }
        try {
            byte[] key = MessageDigest.getInstance("SHA-256")
                    .digest(encryptionKey.getBytes(StandardCharsets.UTF_8));
            return new SecretKeySpec(key, ALGORITHM);
        } catch (Exception e) {
            throw new BusinessException(ResultCode.SYSTEM_EXECUTION_ERROR, "加密密钥派生失败");
        }
    }
}
