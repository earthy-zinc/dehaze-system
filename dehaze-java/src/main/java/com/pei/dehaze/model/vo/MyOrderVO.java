package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonInclude;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@JsonInclude(JsonInclude.Include.NON_NULL)
@Schema(description = "我的订单列表VO")
public class MyOrderVO {

    @Schema(description = "订单ID")
    private Long id;

    @Schema(description = "订单号")
    private String orderNo;

    @Schema(description = "套餐名称")
    private String packageName;

    @Schema(description = "套餐等级")
    private String packageLevel;

    @Schema(description = "应付金额（分）")
    private Long payableAmount;

    @Schema(description = "实付金额（分）")
    private Long paidAmount;

    @Schema(description = "支付方式")
    private String payMethod;

    @Schema(description = "订单状态")
    private String status;

    @Schema(description = "创建时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;

    @Schema(description = "支付时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime paidTime;

    @Schema(description = "套餐到期时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime packageExpireTime;
}
