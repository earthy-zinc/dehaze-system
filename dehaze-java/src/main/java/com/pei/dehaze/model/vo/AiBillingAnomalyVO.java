package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;

/** 计费异常事件 */
@Schema(description = "计费异常事件")
@Data
public class AiBillingAnomalyVO {

    private Long id;

    private Long userId;

    private Long billingId;

    private String anomalyType;

    private String detail;

    /** 0:待处理;1:已处理;2:已忽略 */
    private Integer status;

    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime triggerAt;

    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;
}
