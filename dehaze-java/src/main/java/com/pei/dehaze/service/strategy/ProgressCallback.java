package com.pei.dehaze.service.strategy;

/**
 * 任务进度回调接口
 */
public interface ProgressCallback {

    /**
     * 更新任务进度
     * @param current 当前进度值
     * @param total 总量
     * @param message 进度描述信息
     */
    void updateProgress(int current, int total, String message);

    /**
     * 检查任务是否被取消
     * @return true 表示已取消
     */
    boolean isCancelled();

    /**
     * 检查取消状态，如果已取消则抛出异常
     * @throws TaskCancelledException 任务被取消时抛出
     */
    default void checkCancelled() {
        if (isCancelled()) {
            throw new TaskCancelledException("任务已被取消");
        }
    }
}
