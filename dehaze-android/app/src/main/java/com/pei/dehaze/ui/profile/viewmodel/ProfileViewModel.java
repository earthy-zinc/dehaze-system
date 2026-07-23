package com.pei.dehaze.ui.profile.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.pei.dehaze.repository.ProfileRepository;
import com.pei.dehaze.repository.RepositoryCallback;
import com.pei.dehaze.ui.common.BaseViewModel;
import com.pei.dehaze.sdk.model.user.UserInfo;

public class ProfileViewModel extends BaseViewModel {

    private final ProfileRepository profileRepository;

    private final MutableLiveData<UserInfo> userInfo = new MutableLiveData<>();
    private final MutableLiveData<Boolean> logoutSuccess = new MutableLiveData<>(false);

    public ProfileViewModel() {
        profileRepository = new ProfileRepository();
    }

    public void loadUserInfo() {
        profileRepository.getUserInfo(withLoading(userInfo::postValue));
    }

    public void logout() {
        profileRepository.logout(withLoading(v -> logoutSuccess.postValue(true)));
    }

    public LiveData<UserInfo> getUserInfo() {
        return userInfo;
    }

    public LiveData<Boolean> getLogoutSuccess() {
        return logoutSuccess;
    }
}
