package com.pei.dehaze.model.read;

import lombok.Data;

import java.math.BigDecimal;

/** 性能趋势聚合行（按维度+日期） */
@Data
public class AiTraceTrendRead {

    private String dimension;

    private String date;

    private Long callCount;

    private Long successCount;

    /** 首 Token 延迟仅取成功调用口径 */
    private BigDecimal avgFirstTokenMs;

    private BigDecimal avgDurationMs;
}
