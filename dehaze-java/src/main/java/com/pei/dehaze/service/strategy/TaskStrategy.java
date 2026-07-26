package com.pei.dehaze.service.strategy;

import com.pei.dehaze.model.entity.SysTask;
import java.util.List;
import java.util.Map;

/**
 * 任务策略接口
 * 定义任务执行的统一契约
 */
public interface TaskStrategy {

    /**
     * 获取策略支持的任务类型列表
     * <p>支持一个策略处理多个任务类型（如 GenericExportStrategy 处理所有 *_export 类型）
     * @return 任务类型列表（如：dataset_export、user_export、user_import）
     */
    List<String> getTaskTypes();

    /**
     * 执行任务
     * @param task 任务实体
     * @param params 任务参数
     * @param callback 进度回调接口
     * @return 任务结果
     */
    TaskResult execute(SysTask task, Map<String, Object> params, ProgressCallback callback);

    /**
     * 取消任务（清理资源）
     * @param task 任务实体
     */
    default void cancel(SysTask task) {
    }

    /**
     * 验证任务参数
     * @param params 任务参数
     */
    default void validateParams(Map<String, Object> params) {
    }
}
