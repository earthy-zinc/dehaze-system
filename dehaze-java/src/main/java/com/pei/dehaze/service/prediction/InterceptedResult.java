package com.pei.dehaze.service.prediction;

import lombok.Builder;
import lombok.Data;

/**
 * 拦截器命中后返回的结果，主流程据此直接写 completed 日志并返回前端
 */
@Data
@Builder
public class InterceptedResult {

    /**
     * 处理结果图片 URL
     */
    private final String resultUrl;

    /**
     * 处理结果图片 MD5
     */
    private final String resultMd5;

    /**
     * 处理结果图片对应的 sys_file.id（若结果已入库）
     */
    private final Long resultFileId;
}
