package com.pei.dehaze.repository;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.api.AlgorithmAPI;
import com.pei.dehaze.sdk.api.DatasetAPI;
import com.pei.dehaze.sdk.api.ModelAPI;
import com.pei.dehaze.sdk.api.TaskAPI;
import com.pei.dehaze.sdk.api.UserAPI;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.algorithm.Algorithm;
import com.pei.dehaze.sdk.model.algorithm.AlgorithmQuery;
import com.pei.dehaze.sdk.model.dataset.Dataset;
import com.pei.dehaze.sdk.model.dataset.DatasetQuery;
import com.pei.dehaze.sdk.model.prediction.PredictionLogVO;
import com.pei.dehaze.sdk.model.task.TaskQuery;
import com.pei.dehaze.sdk.model.task.TaskVO;
import com.pei.dehaze.sdk.model.user.UserInfo;
import com.pei.dehaze.sdk.network.ApiException;

import java.util.ArrayList;
import java.util.List;

public class DashboardRepository {

    public interface UserInfoCallback {
        void onSuccess(UserInfo userInfo);
        void onError(String errorMessage);
    }

    public interface StatsCallback {
        void onSuccess(StatsData stats);
        void onError(String errorMessage);
    }

    public interface RecentActivitiesCallback {
        void onSuccess(List<PredictionLogVO> activities);
        void onError(String errorMessage);
    }

    public static class StatsData {
        private final long datasetCount;
        private final long algorithmCount;
        private final long taskCount;
        private final long historyCount;

        public StatsData(long datasetCount, long algorithmCount, long taskCount, long historyCount) {
            this.datasetCount = datasetCount;
            this.algorithmCount = algorithmCount;
            this.taskCount = taskCount;
            this.historyCount = historyCount;
        }

        public long getDatasetCount() {
            return datasetCount;
        }

        public long getAlgorithmCount() {
            return algorithmCount;
        }

        public long getTaskCount() {
            return taskCount;
        }

        public long getHistoryCount() {
            return historyCount;
        }
    }

    public void getUserInfo(UserInfoCallback callback) {
        UserAPI.getInfo(new ApiCallback<UserInfo>() {
            @Override
            public void onSuccess(UserInfo data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }

    public void getStats(StatsCallback callback) {
        StatsData[] stats = new StatsData[1];
        final long[] pending = {4};
        final String[] firstError = {null};

        Runnable checkComplete = () -> {
            if (pending[0] == 0) {
                if (firstError[0] != null) {
                    callback.onError(firstError[0]);
                } else {
                    callback.onSuccess(stats[0] != null ? stats[0]
                            : new StatsData(0, 0, 0, 0));
                }
            }
        };

        DatasetQuery datasetQuery = new DatasetQuery();
        datasetQuery.setPageNum(1);
        datasetQuery.setPageSize(1);
        DatasetAPI.getList(datasetQuery, new ApiCallback<PageResult<Dataset>>() {
            @Override
            public void onSuccess(PageResult<Dataset> data) {
                long datasetCount = data.getTotal();
                mergeStats(stats, pending, firstError, checkComplete, 0, datasetCount);
            }

            @Override
            public void onError(String code, String message) {
                if (firstError[0] == null) {
                    firstError[0] = "[" + code + "] " + message;
                }
                synchronized (pending) {
                    pending[0]--;
                }
                checkComplete.run();
            }

            @Override
            public void onFailure(ApiException e) {
                if (firstError[0] == null) {
                    firstError[0] = e.getMessage();
                }
                synchronized (pending) {
                    pending[0]--;
                }
                checkComplete.run();
            }
        });

        AlgorithmAPI.getList(new AlgorithmQuery(), new ApiCallback<List<Algorithm>>() {
            @Override
            public void onSuccess(List<Algorithm> data) {
                long algorithmCount = data != null ? data.size() : 0;
                mergeStats(stats, pending, firstError, checkComplete, 1, algorithmCount);
            }

            @Override
            public void onError(String code, String message) {
                if (firstError[0] == null) {
                    firstError[0] = "[" + code + "] " + message;
                }
                synchronized (pending) {
                    pending[0]--;
                }
                checkComplete.run();
            }

            @Override
            public void onFailure(ApiException e) {
                if (firstError[0] == null) {
                    firstError[0] = e.getMessage();
                }
                synchronized (pending) {
                    pending[0]--;
                }
                checkComplete.run();
            }
        });

        TaskQuery taskQuery = new TaskQuery();
        taskQuery.setPageNum(1);
        taskQuery.setPageSize(1);
        TaskAPI.getTaskPage(taskQuery, new ApiCallback<PageResult<TaskVO>>() {
            @Override
            public void onSuccess(PageResult<TaskVO> data) {
                long taskCount = data.getTotal();
                mergeStats(stats, pending, firstError, checkComplete, 2, taskCount);
            }

            @Override
            public void onError(String code, String message) {
                if (firstError[0] == null) {
                    firstError[0] = "[" + code + "] " + message;
                }
                synchronized (pending) {
                    pending[0]--;
                }
                checkComplete.run();
            }

            @Override
            public void onFailure(ApiException e) {
                if (firstError[0] == null) {
                    firstError[0] = e.getMessage();
                }
                synchronized (pending) {
                    pending[0]--;
                }
                checkComplete.run();
            }
        });

        ModelAPI.listPredictionLogs(null, 1, 1, new ApiCallback<PageResult<PredictionLogVO>>() {
            @Override
            public void onSuccess(PageResult<PredictionLogVO> data) {
                long historyCount = data.getTotal();
                mergeStats(stats, pending, firstError, checkComplete, 3, historyCount);
            }

            @Override
            public void onError(String code, String message) {
                if (firstError[0] == null) {
                    firstError[0] = "[" + code + "] " + message;
                }
                synchronized (pending) {
                    pending[0]--;
                }
                checkComplete.run();
            }

            @Override
            public void onFailure(ApiException e) {
                if (firstError[0] == null) {
                    firstError[0] = e.getMessage();
                }
                synchronized (pending) {
                    pending[0]--;
                }
                checkComplete.run();
            }
        });
    }

    private void mergeStats(StatsData[] stats, long[] pending, String[] firstError,
                           Runnable checkComplete, int index, long value) {
        synchronized (stats) {
            long[] values = new long[4];
            if (stats[0] != null) {
                values[0] = stats[0].getDatasetCount();
                values[1] = stats[0].getAlgorithmCount();
                values[2] = stats[0].getTaskCount();
                values[3] = stats[0].getHistoryCount();
            }
            values[index] = value;
            stats[0] = new StatsData(values[0], values[1], values[2], values[3]);
        }
        synchronized (pending) {
            pending[0]--;
        }
        checkComplete.run();
    }

    public void getRecentActivities(RecentActivitiesCallback callback) {
        ModelAPI.listPredictionLogs(null, 1, 10, new ApiCallback<PageResult<PredictionLogVO>>() {
            @Override
            public void onSuccess(PageResult<PredictionLogVO> data) {
                callback.onSuccess(data.getList() != null ? data.getList() : new ArrayList<>());
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }
}
