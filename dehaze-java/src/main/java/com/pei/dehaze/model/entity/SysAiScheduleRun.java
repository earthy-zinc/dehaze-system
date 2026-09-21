package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.io.Serial;
import java.io.Serializable;
import java.math.BigDecimal;
import java.time.LocalDateTime;

/**
 * AI 定时任务执行历史（只追加）
 *
 * @author dehaze
 */
@Data
@TableName("sys_ai_schedule_run")
public class SysAiScheduleRun implements Serializable {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long scheduleId;

    private Long userId;

    private LocalDateTime windowStart;

    private Integer status;

    private String skipReason;

    private BigDecimal credits;

    private Integer durationMs;

    private String errorMsg;

    private Long conversationId;

    private String requestId;

    private LocalDateTime createTime;

    @Serial
    private static final long serialVersionUID = 1L;
}
