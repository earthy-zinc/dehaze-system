package com.pei.dehaze.repository;

import com.pei.dehaze.sdk.api.AuthAPI;
import com.pei.dehaze.sdk.model.auth.CaptchaResponse;
import com.pei.dehaze.sdk.model.auth.LoginRequest;
import com.pei.dehaze.sdk.model.auth.LoginResponse;

public class AuthRepository {

    public void getCaptcha(RepositoryCallback<CaptchaResponse> callback) {
        AuthAPI.getCaptcha(RepositoryAdapters.wrap(callback));
    }

    public void login(LoginRequest request, RepositoryCallback<LoginResponse> callback) {
        AuthAPI.login(request, RepositoryAdapters.wrap(callback));
    }

    public void register(LoginRequest request, RepositoryCallback<LoginResponse> callback) {
        AuthAPI.register(request, RepositoryAdapters.wrap(callback));
    }
}
