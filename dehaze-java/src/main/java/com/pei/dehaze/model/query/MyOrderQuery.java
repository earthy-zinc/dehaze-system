package com.pei.dehaze.model.query;

import com.pei.dehaze.common.base.BasePageQuery;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@EqualsAndHashCode(callSuper = true)
@Schema(description = "我的订单查询参数")
public class MyOrderQuery extends BasePageQuery {

    @Schema(description = "订单状态(pending/paid/completed/cancelled/refunding/refunded)")
    private String status;
}
