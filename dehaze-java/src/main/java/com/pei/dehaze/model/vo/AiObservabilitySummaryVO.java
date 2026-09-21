package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

/** 可观测性异常总览统计 */
@Schema(description = "可观测性异常总览统计")
@Data
public class AiObservabilitySummaryVO {

    private Long total;

    private Long successCount;

    private Long failedCount;

    private Long interruptedCount;

    private Long timeoutCount;

    /** 按采集链路写入的拒绝类 error_type 统计 */
    private Long quotaRejected;

    /** 推理步数超阈值或存在失败的工具调用 */
    private Long highRiskCalls;
}
