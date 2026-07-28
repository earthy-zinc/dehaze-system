package com.pei.dehaze.model.query;

import com.pei.dehaze.common.base.BasePageQuery;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@EqualsAndHashCode(callSuper = true)
@Schema(description = "优惠券分页查询参数")
public class CouponPageQuery extends BasePageQuery {

    @Schema(description = "优惠券名称")
    private String name;

    @Schema(description = "类型(full_reduction/discount/no_threshold/trial)")
    private String type;

    @Schema(description = "状态(1:启用;0:禁用)")
    private Integer status;
}
