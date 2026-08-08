package com.pei.dehaze.ui.dashboard.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.pei.dehaze.repository.DashboardRepository;
import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.repository.RepositoryCallback;
import com.pei.dehaze.sdk.api.ModelAPI;
import com.pei.dehaze.sdk.api.UserAPI;
import com.pei.dehaze.ui.common.BaseViewModel;
import com.pei.dehaze.sdk.model.prediction.PredictionLogVO;
import com.pei.dehaze.sdk.model.user.UserInfo;

import java.util.ArrayList;
import java.util.List;

public class DashboardViewModel extends BaseViewModel {

    private final DashboardRepository dashboardRepository = new DashboardRepository();

    private final MutableLiveData<UserInfo> userInfo = new MutableLiveData<>();
    private final MutableLiveData<DashboardRepository.StatsData> stats = new MutableLiveData<>();
    private final MutableLiveData<List<PredictionLogVO>> recentActivities = new MutableLiveData<>();
    private final MutableLiveData<DashboardRepository.StatusDistributionData> statusDistribution = new MutableLiveData<>();
    private final MutableLiveData<List<DashboardRepository.TrendItem>> taskTrend = new MutableLiveData<>();

    public void loadUserInfo() {
        UserAPI.getInfo(RepositoryAdapters.wrap(withLoading(userInfo::postValue)));
    }

    public void loadStats() {
        dashboardRepository.getStats(withLoading(stats::postValue));
    }

    public void loadRecentActivities() {
        ModelAPI.listPredictionLogs(null, 1, 10, RepositoryAdapters.wrapPage(new RepositoryCallback<List<PredictionLogVO>>() {
            @Override
            public void onSuccess(List<PredictionLogVO> activities) {
                recentActivities.postValue(activities != null ? activities : new ArrayList<>());
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
            }
        }));
    }

    public void loadStatusDistribution() {
        dashboardRepository.getTaskStatusDistribution(new RepositoryCallback<DashboardRepository.StatusDistributionData>() {
            @Override
            public void onSuccess(DashboardRepository.StatusDistributionData data) {
                statusDistribution.postValue(data);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
            }
        });
    }

    public void loadTaskTrend() {
        dashboardRepository.getTaskTrend(new RepositoryCallback<List<DashboardRepository.TrendItem>>() {
            @Override
            public void onSuccess(List<DashboardRepository.TrendItem> data) {
                taskTrend.postValue(data);
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
        loadStatusDistribution();
        loadTaskTrend();
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

    public LiveData<DashboardRepository.StatusDistributionData> getStatusDistribution() {
        return statusDistribution;
    }

    public LiveData<List<DashboardRepository.TrendItem>> getTaskTrend() {
        return taskTrend;
    }
}
