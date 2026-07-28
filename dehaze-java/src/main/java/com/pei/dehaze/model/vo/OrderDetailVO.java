package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;

import java.time.LocalDateTime;
import java.util.List;

@Data
@EqualsAndHashCode(callSuper = true)
@Schema(description = "订单详情VO")
public class OrderDetailVO extends OrderPageVO {

    @Schema(description = "订单超时时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime expireTime;

    @Schema(description = "权益生效时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime effectiveTime;

    @Schema(description = "取消原因")
    private String cancelReason;

    @Schema(description = "是否自动续费(0:否;1:是)")
    private Integer isAutoRenew;

    @Schema(description = "支付流水列表")
    private List<PaymentRecordVO> paymentRecords;

    @Schema(description = "退款记录")
    private RefundRecordVO refundRecord;
}
