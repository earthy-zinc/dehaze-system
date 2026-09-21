package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Data;

import java.math.BigDecimal;
import java.time.LocalDateTime;

/**
 * 定时任务执行历史项
 *
 * @author dehaze
 */
@Data
public class AiScheduleHistoryVO {

    private Long id;

    private Long scheduleId;

    private Integer status;

    private String skipReason;

    private BigDecimal credits;

    private Integer durationMs;

    private String errorMsg;

    private Long conversationId;

    private String requestId;

    @JsonInclude(JsonInclude.Include.NON_NULL)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime windowStart;

    @JsonInclude(JsonInclude.Include.NON_NULL)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;
}
