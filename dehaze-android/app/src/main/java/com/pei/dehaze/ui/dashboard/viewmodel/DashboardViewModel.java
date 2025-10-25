package com.pei.dehaze.ui.dashboard.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import com.pei.dehaze.repository.DashboardRepository;
import com.pei.dehaze.sdk.model.user.UserInfo;

public class DashboardViewModel extends ViewModel {

    private final DashboardRepository dashboardRepository;

    private final MutableLiveData<UserInfo> userInfo = new MutableLiveData<>();
    private final MutableLiveData<Boolean> loading = new MutableLiveData<>();
    private final MutableLiveData<String> error = new MutableLiveData<>();

    public DashboardViewModel() {
        dashboardRepository = new DashboardRepository();
    }

    public void loadUserInfo() {
        loading.setValue(true);
        dashboardRepository.getUserInfo(new DashboardRepository.UserInfoCallback() {
            @Override
            public void onSuccess(UserInfo userInfoResponse) {
                userInfo.postValue(userInfoResponse);
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
}