package com.pei.dehaze.service.strategy;

import lombok.Builder;
import lombok.Data;
import java.util.Map;

/**
 * 任务执行结果
 */
@Data
@Builder
public class TaskResult {

    /** 是否成功 */
    private boolean success;

    /** 结果数据（如下载 URL） */
    private String data;

    /** 错误信息 */
    private String errorMessage;

    /** 附加元数据 */
    private Map<String, Object> metadata;

    /**
     * 创建成功结果
     */
    public static TaskResult success(String data) {
        return TaskResult.builder()
                .success(true)
                .data(data)
                .build();
    }

    /**
     * 创建成功结果（带元数据）
     */
    public static TaskResult success(String data, Map<String, Object> metadata) {
        return TaskResult.builder()
                .success(true)
                .data(data)
                .metadata(metadata)
                .build();
    }

    /**
     * 创建失败结果
     */
    public static TaskResult failure(String errorMessage) {
        return TaskResult.builder()
                .success(false)
                .errorMessage(errorMessage)
                .build();
    }
}
