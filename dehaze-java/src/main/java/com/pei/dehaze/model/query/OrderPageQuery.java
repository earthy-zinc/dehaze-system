package com.pei.dehaze.model.query;

import com.pei.dehaze.common.base.BasePageQuery;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;

import java.time.LocalDateTime;

@Data
@EqualsAndHashCode(callSuper = true)
@Schema(description = "订单后台分页查询参数")
public class OrderPageQuery extends BasePageQuery {

    @Schema(description = "订单号")
    private String orderNo;

    @Schema(description = "用户名/昵称关键字")
    private String keywords;

    @Schema(description = "订单状态")
    private String status;

    @Schema(description = "支付方式")
    private String payMethod;

    @Schema(description = "金额下限（分）")
    private Long amountMin;

    @Schema(description = "金额上限（分）")
    private Long amountMax;

    @Schema(description = "支付时间起")
    private LocalDateTime paidTimeStart;

    @Schema(description = "支付时间止")
    private LocalDateTime paidTimeEnd;
}
