package com.pei.dehaze.service;

import com.pei.dehaze.model.dto.ApiKeyResult;
import com.pei.dehaze.model.form.ApiKeyForm;
import org.springframework.security.core.Authentication;

import java.util.List;

/**
 * API 密钥服务接口
 * <p>
 * 吊销机制：使用 {@link #revokeApiKey(Long)} 设 revoked_at，绝不再用物理删除表示吊销。
 * </p>
 */
public interface ApiKeyService {

    ApiKeyResult createApiKey(ApiKeyForm form);

    List<ApiKeyResult> listApiKeys();

    /**
     * 吊销 API 密钥（设 revoked_at = now()）
     *
     * @param id 密钥ID
     * @return 成功则返回 true
     */
    boolean revokeApiKey(Long id);

    Authentication authenticateByKey(String rawKey);
}
