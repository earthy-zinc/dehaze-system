package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Data;

import java.math.BigDecimal;
import java.time.LocalDateTime;

/**
 * 定时任务最近一次执行摘要
 *
 * @author dehaze
 */
@Data
public class AiScheduleRunSummaryVO {

    private Integer status;

    private String skipReason;

    private BigDecimal credits;

    private Integer durationMs;

    private String errorMsg;

    private Long conversationId;

    @JsonInclude(JsonInclude.Include.NON_NULL)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;
}
