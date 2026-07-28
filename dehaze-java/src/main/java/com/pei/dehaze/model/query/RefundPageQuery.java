package com.pei.dehaze.model.query;

import com.pei.dehaze.common.base.BasePageQuery;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;

import java.time.LocalDateTime;

@Data
@EqualsAndHashCode(callSuper = true)
@Schema(description = "退款记录分页查询参数")
public class RefundPageQuery extends BasePageQuery {

    @Schema(description = "订单号")
    private String orderNo;

    @Schema(description = "用户名/昵称关键字")
    private String keywords;

    @Schema(description = "退款状态(refunding/refunded/refund_failed)")
    private String status;

    @Schema(description = "申请时间起")
    private LocalDateTime applyTimeStart;

    @Schema(description = "申请时间止")
    private LocalDateTime applyTimeEnd;
}
