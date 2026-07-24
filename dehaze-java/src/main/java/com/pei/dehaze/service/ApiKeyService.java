package com.pei.dehaze.service;

import com.pei.dehaze.model.dto.ApiKeyResult;
import com.pei.dehaze.model.form.ApiKeyForm;
import org.springframework.security.core.Authentication;

import java.util.List;

public interface ApiKeyService {

    ApiKeyResult createApiKey(ApiKeyForm form);

    List<ApiKeyResult> listApiKeys();

    boolean deleteApiKey(Long id);

    Authentication authenticateByKey(String rawKey);
}
