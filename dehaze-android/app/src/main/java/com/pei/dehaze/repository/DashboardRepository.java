package com.pei.dehaze.repository;

import com.pei.dehaze.sdk.api.AlgorithmAPI;
import com.pei.dehaze.sdk.api.DatasetAPI;
import com.pei.dehaze.sdk.api.ModelAPI;
import com.pei.dehaze.sdk.api.TaskAPI;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.algorithm.Algorithm;
import com.pei.dehaze.sdk.model.algorithm.AlgorithmQuery;
import com.pei.dehaze.sdk.model.dataset.Dataset;
import com.pei.dehaze.sdk.model.dataset.DatasetQuery;
import com.pei.dehaze.sdk.model.prediction.PredictionLogVO;
import com.pei.dehaze.sdk.model.task.TaskQuery;
import com.pei.dehaze.sdk.model.task.TaskStatus;
import com.pei.dehaze.sdk.model.task.TaskVO;

import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
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

    /**
     * 任务状态分布数据
     */
    public static class StatusDistributionData {
        private final Map<TaskStatus, Long> distribution;

        public StatusDistributionData(Map<TaskStatus, Long> distribution) {
            this.distribution = distribution;
        }

        public Map<TaskStatus, Long> getDistribution() {
            return distribution;
        }
    }

    /**
     * 任务趋势数据（单日计数）
     */
    public static class TrendItem {
        private final String date;
        private final long count;

        public TrendItem(String date, long count) {
            this.date = date;
            this.count = count;
        }

        public String getDate() {
            return date;
        }

        public long getCount() {
            return count;
        }
    }

    /**
     * 获取任务状态分布统计。
     * 按每种状态分别发起分页查询（pageSize=1），从 PageResult.total 获取各状态数量。
     */
    public void getTaskStatusDistribution(RepositoryCallback<StatusDistributionData> callback) {
        TaskStatus[] statuses = TaskStatus.values();
        Map<TaskStatus, Long> distribution = new HashMap<>();
        AtomicInteger pending = new AtomicInteger(statuses.length);
        AtomicReference<String> firstError = new AtomicReference<>(null);

        for (TaskStatus status : statuses) {
            TaskQuery query = new TaskQuery();
            query.setStatus(status);
            query.setPageNum(1);
            query.setPageSize(1);
            TaskAPI.getTaskPage(query, RepositoryAdapters.wrap(new RepositoryCallback<PageResult<TaskVO>>() {
                @Override
                public void onSuccess(PageResult<TaskVO> data) {
                    synchronized (distribution) {
                        distribution.put(status, data != null ? data.getTotal() : 0L);
                    }
                    onDistComplete();
                }

                @Override
                public void onError(String errorMessage) {
                    synchronized (distribution) {
                        distribution.put(status, 0L);
                    }
                    firstError.compareAndSet(null, errorMessage);
                    onDistComplete();
                }

                private void onDistComplete() {
                    if (pending.decrementAndGet() == 0) {
                        String err = firstError.get();
                        if (err != null) {
                            callback.onError(err);
                        } else {
                            callback.onSuccess(new StatusDistributionData(distribution));
                        }
                    }
                }
            }));
        }
    }

    /**
     * 获取近 7 天任务创建趋势。
     * 查询最近 7 天每一天的任务总数，通过分页查询（pageSize=1）获取 total。
     * 日期按 createdAt 字段截取日期部分进行聚合。
     */
    public void getTaskTrend(RepositoryCallback<List<TrendItem>> callback) {
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd", Locale.getDefault());
        Calendar cal = Calendar.getInstance();
        String today = sdf.format(cal.getTime());
        cal.add(Calendar.DAY_OF_YEAR, -6);
        String sevenDaysAgo = sdf.format(cal.getTime());

        // 按创建时间范围查询：后端 TaskQuery 没有日期筛选，这里查询全量后按 createdAt 聚合
        // 先查大分页获取所有任务，再按 createdAt 日期聚合
        TaskQuery query = new TaskQuery();
        query.setPageNum(1);
        query.setPageSize(1000); // 足够大的 pageSize 拉取近期任务
        TaskAPI.getTaskPage(query, RepositoryAdapters.wrap(new RepositoryCallback<PageResult<TaskVO>>() {
            @Override
            public void onSuccess(PageResult<TaskVO> data) {
                List<TaskVO> tasks = data != null ? data.getList() : null;
                // 按日期聚合
                Map<String, Long> dateCount = new HashMap<>();
                // 初始化 7 天
                Calendar initCal = Calendar.getInstance();
                for (int i = 6; i >= 0; i--) {
                    initCal.setTime(new Date());
                    initCal.add(Calendar.DAY_OF_YEAR, -i);
                    dateCount.put(sdf.format(initCal.getTime()), 0L);
                }

                if (tasks != null) {
                    for (TaskVO task : tasks) {
                        String createdAt = task.getCreatedAt();
                        if (createdAt != null && createdAt.length() >= 10) {
                            String datePart = createdAt.substring(0, 10);
                            if (dateCount.containsKey(datePart)) {
                                dateCount.put(datePart, dateCount.get(datePart) + 1);
                            }
                        }
                    }
                }

                List<TrendItem> trend = new java.util.ArrayList<>();
                Calendar iterCal = Calendar.getInstance();
                iterCal.add(Calendar.DAY_OF_YEAR, -6);
                for (int i = 0; i < 7; i++) {
                    String date = sdf.format(iterCal.getTime());
                    trend.add(new TrendItem(date, dateCount.getOrDefault(date, 0L)));
                    iterCal.add(Calendar.DAY_OF_YEAR, 1);
                }
                callback.onSuccess(trend);
            }

            @Override
            public void onError(String errorMessage) {
                callback.onError(errorMessage);
            }
        }));
    }
}
