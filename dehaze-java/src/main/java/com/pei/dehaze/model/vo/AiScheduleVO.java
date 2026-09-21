package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.Map;

/**
 * 定时任务响应（lastRun 仅列表聚合返回）
 *
 * @author dehaze
 */
@Data
public class AiScheduleVO {

    private Long id;

    private Long userId;

    private String name;

    private String cron;

    private String timezone;

    private Map<String, Object> input;

    private Map<String, Object> output;

    private Integer enabled;

    private Integer status;

    private Integer circuitStreak;

    @JsonInclude(JsonInclude.Include.NON_NULL)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime nextTriggerTime;

    @JsonInclude(JsonInclude.Include.NON_NULL)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;

    private AiScheduleRunSummaryVO lastRun;
}
