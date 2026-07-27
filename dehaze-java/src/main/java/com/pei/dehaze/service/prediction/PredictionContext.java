package com.pei.dehaze.service.prediction;

import com.pei.dehaze.model.entity.SysAlgorithm;
import com.pei.dehaze.model.entity.SysFile;
import lombok.Builder;
import lombok.Data;

/**
 * 预测请求上下文，供拦截器读取
 */
@Data
@Builder
public class PredictionContext {

    private final SysAlgorithm algorithm;

    private final Long fileId;

    private final String imageUrl;

    /**
     * 通过 fileId 查询到的原始文件实体；fileId 为空时为 null
     */
    private final SysFile originFile;

    private final String params;

    private final long startTimeMs;
}
