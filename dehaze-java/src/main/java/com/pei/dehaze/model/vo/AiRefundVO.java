package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;

/** AI 积分误扣退款申请 */
@Schema(description = "AI 积分误扣退款申请")
@Data
public class AiRefundVO {

    private Long id;

    private Long userId;

    private Long billingId;

    private Integer amount;

    private String reason;

    /** 1:待审核;2:已通过;3:已驳回 */
    private Integer status;

    private Long auditorId;

    private String auditRemark;

    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;

    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime updateTime;
}
