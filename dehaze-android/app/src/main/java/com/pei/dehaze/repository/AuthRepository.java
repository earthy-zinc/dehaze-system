package com.pei.dehaze.repository;

import com.pei.dehaze.sdk.api.AuthAPI;
import com.pei.dehaze.sdk.model.auth.CaptchaResponse;
import com.pei.dehaze.sdk.model.auth.LoginRequest;
import com.pei.dehaze.sdk.model.auth.LoginResponse;

/**
 * 认证 Repository，封装 AuthAPI 调用，供 LoginViewModel 使用。
 * <p>
 * 注：logout 已在 ProfileRepository 中包装，此处不再重复。
 */
public class AuthRepository {

    public void getCaptcha(RepositoryCallback<CaptchaResponse> callback) {
        AuthAPI.getCaptcha(RepositoryAdapters.wrap(callback));
    }

    public void login(LoginRequest request, RepositoryCallback<LoginResponse> callback) {
        AuthAPI.login(request, RepositoryAdapters.wrap(callback));
    }
}
