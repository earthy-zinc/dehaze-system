package com.pei.dehaze.ui.dashboard.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import com.pei.dehaze.repository.DashboardRepository;
import com.pei.dehaze.sdk.model.prediction.PredictionLogVO;
import com.pei.dehaze.sdk.model.user.UserInfo;

import java.util.ArrayList;
import java.util.List;

public class DashboardViewModel extends ViewModel {

    private final DashboardRepository dashboardRepository;

    private final MutableLiveData<UserInfo> userInfo = new MutableLiveData<>();
    private final MutableLiveData<DashboardRepository.StatsData> stats = new MutableLiveData<>();
    private final MutableLiveData<List<PredictionLogVO>> recentActivities = new MutableLiveData<>();
    private final MutableLiveData<Boolean> loading = new MutableLiveData<>(false);
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

    public void loadStats() {
        loading.setValue(true);
        dashboardRepository.getStats(new DashboardRepository.StatsCallback() {
            @Override
            public void onSuccess(DashboardRepository.StatsData statsData) {
                stats.postValue(statsData);
                loading.postValue(false);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void loadRecentActivities() {
        dashboardRepository.getRecentActivities(new DashboardRepository.RecentActivitiesCallback() {
            @Override
            public void onSuccess(List<PredictionLogVO> activities) {
                recentActivities.postValue(activities != null ? activities : new ArrayList<>());
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
            }
        });
    }

    public void refresh() {
        loadUserInfo();
        loadStats();
        loadRecentActivities();
    }

    public LiveData<UserInfo> getUserInfo() {
        return userInfo;
    }

    public LiveData<DashboardRepository.StatsData> getStats() {
        return stats;
    }

    public LiveData<List<PredictionLogVO>> getRecentActivities() {
        return recentActivities;
    }

    public LiveData<Boolean> getLoading() {
        return loading;
    }

    public LiveData<String> getError() {
        return error;
    }

    public void clearError() {
        error.setValue(null);
    }
}
