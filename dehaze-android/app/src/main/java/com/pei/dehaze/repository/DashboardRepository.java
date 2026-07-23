package com.pei.dehaze.repository;

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

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

public class DashboardRepository {

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

    public void getUserInfo(RepositoryCallback<UserInfo> callback) {
        UserAPI.getInfo(RepositoryAdapters.wrap(callback));
    }

    public void getStats(RepositoryCallback<StatsData> callback) {
        AtomicInteger pending = new AtomicInteger(4);
        AtomicReference<String> firstError = new AtomicReference<>(null);
        long[] counts = new long[4];

        Runnable onComplete = () -> {
            if (pending.decrementAndGet() == 0) {
                String err = firstError.get();
                if (err != null) {
                    callback.onError(err);
                } else {
                    callback.onSuccess(new StatsData(counts[0], counts[1], counts[2], counts[3]));
                }
            }
        };

        DatasetQuery datasetQuery = new DatasetQuery();
        datasetQuery.setPageNum(1);
        datasetQuery.setPageSize(1);
        DatasetAPI.getList(datasetQuery, RepositoryAdapters.wrap(new RepositoryCallback<PageResult<Dataset>>() {
            @Override
            public void onSuccess(PageResult<Dataset> data) {
                counts[0] = data.getTotal();
                onComplete.run();
            }

            @Override
            public void onError(String errorMessage) {
                firstError.compareAndSet(null, errorMessage);
                onComplete.run();
            }
        }));

        AlgorithmAPI.getList(new AlgorithmQuery(), RepositoryAdapters.wrap(new RepositoryCallback<List<Algorithm>>() {
            @Override
            public void onSuccess(List<Algorithm> data) {
                counts[1] = countAlgorithms(data);
                onComplete.run();
            }

            @Override
            public void onError(String errorMessage) {
                firstError.compareAndSet(null, errorMessage);
                onComplete.run();
            }
        }));

        TaskQuery taskQuery = new TaskQuery();
        taskQuery.setPageNum(1);
        taskQuery.setPageSize(1);
        TaskAPI.getTaskPage(taskQuery, RepositoryAdapters.wrap(new RepositoryCallback<PageResult<TaskVO>>() {
            @Override
            public void onSuccess(PageResult<TaskVO> data) {
                counts[2] = data.getTotal();
                onComplete.run();
            }

            @Override
            public void onError(String errorMessage) {
                firstError.compareAndSet(null, errorMessage);
                onComplete.run();
            }
        }));

        ModelAPI.listPredictionLogs(null, 1, 1, RepositoryAdapters.wrap(new RepositoryCallback<PageResult<PredictionLogVO>>() {
            @Override
            public void onSuccess(PageResult<PredictionLogVO> data) {
                counts[3] = data.getTotal();
                onComplete.run();
            }

            @Override
            public void onError(String errorMessage) {
                firstError.compareAndSet(null, errorMessage);
                onComplete.run();
            }
        }));
    }

    /**
     * 递归统计算法树中所有节点数量（含子节点）
     */
    private long countAlgorithms(List<Algorithm> tree) {
        if (tree == null) return 0;
        return tree.stream().mapToLong(a ->
                (a.getChildren() != null ? countAlgorithms(a.getChildren()) : 0) + 1
        ).sum();
    }

    public void getRecentActivities(RepositoryCallback<List<PredictionLogVO>> callback) {
        ModelAPI.listPredictionLogs(null, 1, 10, RepositoryAdapters.wrapPage(callback));
    }
}
