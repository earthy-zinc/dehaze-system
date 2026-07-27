package com.pei.dehaze.ui.profile.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.sdk.api.AuthAPI;
import com.pei.dehaze.sdk.api.UserAPI;
import com.pei.dehaze.sdk.utils.TokenManager;
import com.pei.dehaze.ui.common.BaseViewModel;
import com.pei.dehaze.sdk.model.user.UserInfo;

public class ProfileViewModel extends BaseViewModel {

    private final MutableLiveData<UserInfo> userInfo = new MutableLiveData<>();
    private final MutableLiveData<Boolean> logoutSuccess = new MutableLiveData<>(false);
    private final MutableLiveData<Boolean> notLoggedIn = new MutableLiveData<>(false);

    public void loadUserInfo() {
        if (!TokenManager.hasToken()) {
            notLoggedIn.setValue(true);
            loading.setValue(false);
            return;
        }
        notLoggedIn.setValue(false);
        UserAPI.getInfo(RepositoryAdapters.wrap(withLoading(userInfo::postValue)));
    }

    public void logout() {
        AuthAPI.logout(RepositoryAdapters.wrap(withLoading(v -> logoutSuccess.postValue(true))));
    }

    public LiveData<UserInfo> getUserInfo() {
        return userInfo;
    }

    public LiveData<Boolean> getLogoutSuccess() {
        return logoutSuccess;
    }

    public LiveData<Boolean> getNotLoggedIn() {
        return notLoggedIn;
    }
}
