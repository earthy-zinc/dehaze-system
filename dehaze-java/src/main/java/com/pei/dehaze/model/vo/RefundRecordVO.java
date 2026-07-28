package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonInclude;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@JsonInclude(JsonInclude.Include.NON_NULL)
@Schema(description = "退款记录VO")
public class RefundRecordVO {

    @Schema(description = "退款ID")
    private Long id;

    @Schema(description = "退款单号")
    private String refundNo;

    @Schema(description = "订单ID")
    private Long orderId;

    @Schema(description = "订单号")
    private String orderNo;

    @Schema(description = "用户ID")
    private Long userId;

    @Schema(description = "用户名")
    private String username;

    @Schema(description = "退款金额（分）")
    private Long refundAmount;

    @Schema(description = "退款原因")
    private String reason;

    @Schema(description = "申请时已用权益次数")
    private Integer usedQuota;

    @Schema(description = "退款状态(refunding/refunded/refund_failed)")
    private String status;

    @Schema(description = "退款渠道")
    private String channel;

    @Schema(description = "渠道退款流水号")
    private String channelRefundNo;

    @Schema(description = "申请时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime applyTime;

    @Schema(description = "审核时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime auditTime;

    @Schema(description = "审核人ID")
    private Long auditorId;

    @Schema(description = "审核备注")
    private String auditRemark;

    @Schema(description = "退款完成时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime refundTime;

    @Schema(description = "错误信息")
    private String errorMessage;
}
