package com.pei.dehaze.ui.profile.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import com.pei.dehaze.repository.ProfileRepository;
import com.pei.dehaze.sdk.model.user.UserInfo;

public class ProfileViewModel extends ViewModel {

    private final ProfileRepository profileRepository;

    private final MutableLiveData<UserInfo> userInfo = new MutableLiveData<>();
    private final MutableLiveData<Boolean> loading = new MutableLiveData<>(false);
    private final MutableLiveData<String> error = new MutableLiveData<>();
    private final MutableLiveData<Boolean> logoutSuccess = new MutableLiveData<>(false);

    public ProfileViewModel() {
        profileRepository = new ProfileRepository();
    }

    public void loadUserInfo() {
        loading.setValue(true);
        profileRepository.getUserInfo(new ProfileRepository.UserInfoCallback() {
            @Override
            public void onSuccess(UserInfo data) {
                userInfo.postValue(data);
                loading.postValue(false);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void logout() {
        loading.setValue(true);
        profileRepository.logout(new ProfileRepository.LogoutCallback() {
            @Override
            public void onSuccess() {
                logoutSuccess.postValue(true);
                loading.postValue(false);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public LiveData<UserInfo> getUserInfo() {
        return userInfo;
    }

    public LiveData<Boolean> getLoading() {
        return loading;
    }

    public LiveData<String> getError() {
        return error;
    }

    public LiveData<Boolean> getLogoutSuccess() {
        return logoutSuccess;
    }

    public void clearError() {
        error.setValue(null);
    }
}
