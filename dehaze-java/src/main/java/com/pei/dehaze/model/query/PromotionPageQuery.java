package com.pei.dehaze.model.query;

import com.pei.dehaze.common.base.BasePageQuery;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;

import java.time.LocalDateTime;

@Data
@EqualsAndHashCode(callSuper = true)
@Schema(description = "促销活动分页查询参数")
public class PromotionPageQuery extends BasePageQuery {

    @Schema(description = "活动名称（模糊）")
    private String name;

    @Schema(description = "活动类型(discount/new_user/holiday/full_reduction)")
    private String type;

    @Schema(description = "状态(1:启用;0:禁用)")
    private Integer status;

    @Schema(description = "活动开始时间起")
    private LocalDateTime startTime;

    @Schema(description = "活动结束时间止")
    private LocalDateTime endTime;
}
