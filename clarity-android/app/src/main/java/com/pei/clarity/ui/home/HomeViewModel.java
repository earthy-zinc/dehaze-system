package com.pei.clarity.ui.home;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.pei.clarity.repository.RepositoryAdapters;
import com.pei.clarity.repository.RepositoryCallback;
import com.pei.clarity.sdk.api.AiImageAPI;
import com.pei.clarity.sdk.api.AiModelAPI;
import com.pei.clarity.sdk.api.FavoriteAPI;
import com.pei.clarity.sdk.api.UserAPI;
import com.pei.clarity.sdk.model.PageResult;
import com.pei.clarity.sdk.model.ai_model.AiModelVO;
import com.pei.clarity.sdk.model.favorite.FavoriteCountVO;
import com.pei.clarity.sdk.model.image_task.ImageTaskQuery;
import com.pei.clarity.sdk.model.image_task.ImageTaskVO;
import com.pei.clarity.sdk.model.user.UserInfo;
import com.pei.clarity.ui.common.BaseViewModel;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

public class HomeViewModel extends BaseViewModel {

    private final MutableLiveData<UserInfo> userInfo = new MutableLiveData<>();
    private final MutableLiveData<HomeStats> stats = new MutableLiveData<>();
    private final MutableLiveData<List<ImageTaskVO>> recentActivities = new MutableLiveData<>();

    public void refresh() {
        loadUserInfo();
        loadStats();
        loadRecentActivities();
    }

    private void loadUserInfo() {
        UserAPI.getInfo(RepositoryAdapters.wrap(withLoading(userInfo::postValue)));
    }

    /**
     * 首页统计概览：去雾任务数 / 收藏数 / 可用图像模型数，任一指标失败按 0 计，不阻断其余展示。
     */
    private void loadStats() {
        long[] counts = new long[3];
        AtomicInteger pending = new AtomicInteger(3);
        Runnable finish = () -> {
            if (pending.decrementAndGet() == 0) {
                stats.postValue(new HomeStats(counts[0], counts[1], counts[2]));
            }
        };

        ImageTaskQuery countQuery = new ImageTaskQuery();
        countQuery.setPageNum(1);
        countQuery.setPageSize(1);
        AiImageAPI.listImageTasks(countQuery,
                RepositoryAdapters.wrap(new RepositoryCallback<PageResult<ImageTaskVO>>() {
                    @Override
                    public void onSuccess(PageResult<ImageTaskVO> data) {
                        counts[0] = data == null ? 0 : data.getTotal();
                        finish.run();
                    }

                    @Override
                    public void onError(String errorMessage) {
                        finish.run();
                    }
                }));

        FavoriteAPI.getCount(null, RepositoryAdapters.wrap(new RepositoryCallback<List<FavoriteCountVO>>() {
            @Override
            public void onSuccess(List<FavoriteCountVO> list) {
                if (list != null) {
                    for (FavoriteCountVO item : list) {
                        counts[1] += item.getCount() != null ? item.getCount() : 0;
                    }
                }
                finish.run();
            }

            @Override
            public void onError(String errorMessage) {
                finish.run();
            }
        }));

        AiModelAPI.listEnabledModels("image_edit", null,
                RepositoryAdapters.wrap(new RepositoryCallback<List<AiModelVO>>() {
                    @Override
                    public void onSuccess(List<AiModelVO> list) {
                        counts[2] = list == null ? 0 : list.size();
                        finish.run();
                    }

                    @Override
                    public void onError(String errorMessage) {
                        finish.run();
                    }
                }));
    }

    private void loadRecentActivities() {
        ImageTaskQuery query = new ImageTaskQuery();
        query.setPageNum(1);
        query.setPageSize(5);
        AiImageAPI.listImageTasks(query,
                RepositoryAdapters.wrapPage(new RepositoryCallback<List<ImageTaskVO>>() {
                    @Override
                    public void onSuccess(List<ImageTaskVO> data) {
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

    public LiveData<List<ImageTaskVO>> getRecentActivities() {
        return recentActivities;
    }

    public static class HomeStats {
        private final long processCount;
        private final long favoriteCount;
        private final long modelCount;

        public HomeStats(long processCount, long favoriteCount, long modelCount) {
            this.processCount = processCount;
            this.favoriteCount = favoriteCount;
            this.modelCount = modelCount;
        }

        public long getProcessCount() {
            return processCount;
        }

        public long getFavoriteCount() {
            return favoriteCount;
        }

        public long getModelCount() {
            return modelCount;
        }
    }
}
