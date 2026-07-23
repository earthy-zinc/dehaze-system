package com.pei.dehaze.repository;

import com.pei.dehaze.sdk.api.AuthAPI;
import com.pei.dehaze.sdk.api.UserAPI;
import com.pei.dehaze.sdk.model.user.UserInfo;

public class ProfileRepository {

    public void getUserInfo(RepositoryCallback<UserInfo> callback) {
        UserAPI.getInfo(RepositoryAdapters.wrap(callback));
    }

    public void logout(RepositoryCallback<Void> callback) {
        AuthAPI.logout(RepositoryAdapters.wrap(callback));
    }
}
