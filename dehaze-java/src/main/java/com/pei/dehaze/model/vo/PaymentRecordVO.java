package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonInclude;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@JsonInclude(JsonInclude.Include.NON_NULL)
@Schema(description = "支付流水VO")
public class PaymentRecordVO {

    @Schema(description = "流水ID")
    private Long id;

    @Schema(description = "支付渠道流水号")
    private String paymentNo;

    @Schema(description = "支付渠道")
    private String channel;

    @Schema(description = "支付金额（分）")
    private Long amount;

    @Schema(description = "支付状态(1:处理中;2:成功;3:失败)")
    private Integer status;

    @Schema(description = "回调时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime callbackTime;

    @Schema(description = "创建时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;
}
