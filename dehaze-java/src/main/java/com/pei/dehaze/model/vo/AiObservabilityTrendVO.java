package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

/** 性能趋势行（调用量/成功率/平均延迟，首 Token 延迟取成功调用口径） */
@Schema(description = "性能趋势行")
@Data
public class AiObservabilityTrendVO {

    private String model;

    private String agentCode;

    private String date;

    private Long callCount;

    private Long successCount;

    private Double successRate;

    private Double avgFirstTokenMs;

    private Double avgDurationMs;
}
