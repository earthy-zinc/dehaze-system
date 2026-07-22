package com.pei.dehaze.repository;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.api.AuthAPI;
import com.pei.dehaze.sdk.api.UserAPI;
import com.pei.dehaze.sdk.model.user.UserInfo;
import com.pei.dehaze.sdk.network.ApiException;

public class ProfileRepository {

    public interface UserInfoCallback {
        void onSuccess(UserInfo userInfo);
        void onError(String errorMessage);
    }

    public interface LogoutCallback {
        void onSuccess();
        void onError(String errorMessage);
    }

    public void getUserInfo(UserInfoCallback callback) {
        UserAPI.getInfo(new ApiCallback<UserInfo>() {
            @Override
            public void onSuccess(UserInfo data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }

    public void logout(LogoutCallback callback) {
        AuthAPI.logout(new ApiCallback<Void>() {
            @Override
            public void onSuccess(Void data) {
                callback.onSuccess();
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }
}
