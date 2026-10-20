package com.pei.clarity.service.prediction;

import com.pei.clarity.model.entity.SysFile;
import lombok.Builder;
import lombok.Data;

/**
 * 图像任务请求上下文，供拦截器读取
 */
@Data
@Builder
public class PredictionContext {

    /** 图像模型标识（统一模型规范 slug） */
    private final String modelId;

    private final Long fileId;

    private final String imageUrl;

    /**
     * 通过 fileId 查询到的原始文件实体；fileId 为空时为 null
     */
    private final SysFile originFile;

    private final String params;

    private final long startTimeMs;
}
