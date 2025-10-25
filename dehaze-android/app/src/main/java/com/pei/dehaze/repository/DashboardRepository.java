package com.pei.dehaze.repository;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.api.UserAPI;
import com.pei.dehaze.sdk.model.user.UserInfo;

public class DashboardRepository {

    public interface UserInfoCallback {
        void onSuccess(UserInfo userInfo);
        void onError(String errorMessage);
    }

    public void getUserInfo(UserInfoCallback callback) {
        UserAPI.getInfo(new ApiCallback<UserInfo>() {
            @Override
            public void onSuccess(UserInfo data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(int code, String message) {
                callback.onError("Error " + code + ": " + message);
            }

            @Override
            public void onFailure(com.pei.dehaze.sdk.network.ApiException e) {
                callback.onError("Network error: " + e.getMessage());
            }
        });
    }
}