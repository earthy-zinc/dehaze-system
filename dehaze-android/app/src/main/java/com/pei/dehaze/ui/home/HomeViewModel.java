package com.pei.dehaze.ui.home;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.repository.RepositoryCallback;
import com.pei.dehaze.sdk.api.FavoriteAPI;
import com.pei.dehaze.sdk.api.ModelAPI;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.prediction.PredictionLogVO;
import com.pei.dehaze.sdk.model.user.UserInfo;
import com.pei.dehaze.sdk.api.UserAPI;
import com.pei.dehaze.ui.common.BaseViewModel;

import java.util.ArrayList;
import java.util.List;

public class HomeViewModel extends BaseViewModel {

    private final MutableLiveData<UserInfo> userInfo = new MutableLiveData<>();
    private final MutableLiveData<HomeStats> stats = new MutableLiveData<>();
    private final MutableLiveData<List<PredictionLogVO>> recentActivities = new MutableLiveData<>();

    public void refresh() {
        loadUserInfo();
        loadStats();
        loadRecentActivities();
    }

    private void loadUserInfo() {
        UserAPI.getInfo(RepositoryAdapters.wrap(withLoading(userInfo::postValue)));
    }

    private void loadStats() {
        ModelAPI.listPredictionLogs(null, 1, 1,
                RepositoryAdapters.wrap(new RepositoryCallback<PageResult<PredictionLogVO>>() {
                    @Override
                    public void onSuccess(PageResult<PredictionLogVO> data) {
                        long processCount = data != null ? data.getTotal() : 0;
                        FavoriteAPI.getCount(null,
                                RepositoryAdapters.wrap(new RepositoryCallback<List<com.pei.dehaze.sdk.model.favorite.FavoriteCountVO>>() {
                                    @Override
                                    public void onSuccess(List<com.pei.dehaze.sdk.model.favorite.FavoriteCountVO> favList) {
                                        long favCount = 0;
                                        if (favList != null) {
                                            for (com.pei.dehaze.sdk.model.favorite.FavoriteCountVO f : favList) {
                                                favCount += f.getCount() != null ? f.getCount() : 0;
                                            }
                                        }
                                        stats.postValue(new HomeStats(processCount, favCount));
                                    }

                                    @Override
                                    public void onError(String errorMessage) {
                                        stats.postValue(new HomeStats(processCount, 0));
                                    }
                                }));
                    }

                    @Override
                    public void onError(String errorMessage) {
                        error.postValue(errorMessage);
                    }
                }));
    }

    private void loadRecentActivities() {
        ModelAPI.listPredictionLogs(null, 1, 5,
                RepositoryAdapters.wrapPage(new RepositoryCallback<List<PredictionLogVO>>() {
                    @Override
                    public void onSuccess(List<PredictionLogVO> data) {
                        recentActivities.postValue(data != null ? data : new ArrayList<>());
                    }

                    @Override
                    public void onError(String errorMessage) {
                        error.postValue(errorMessage);
                    }
                }));
    }

    public LiveData<UserInfo> getUserInfo() {
        return userInfo;
    }

    public LiveData<HomeStats> getStats() {
        return stats;
    }

    public LiveData<List<PredictionLogVO>> getRecentActivities() {
        return recentActivities;
    }

    public static class HomeStats {
        private final long processCount;
        private final long favoriteCount;

        public HomeStats(long processCount, long favoriteCount) {
            this.processCount = processCount;
            this.favoriteCount = favoriteCount;
        }

        public long getProcessCount() {
            return processCount;
        }

        public long getFavoriteCount() {
            return favoriteCount;
        }
    }
}
