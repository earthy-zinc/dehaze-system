package com.pei.dehaze.model.query;

import com.pei.dehaze.common.base.BasePageQuery;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;

import java.time.LocalDateTime;

@Data
@EqualsAndHashCode(callSuper = true)
@Schema(description = "套餐分页查询参数")
public class PackagePageQuery extends BasePageQuery {

    @Schema(description = "套餐名称")
    private String name;

    @Schema(description = "会员等级(level_1/level_2/level_3)")
    private String levelCode;

    @Schema(description = "计费周期(monthly/quarterly/yearly)")
    private String period;

    @Schema(description = "状态(1:上架;0:下架)")
    private Integer status;

    @Schema(description = "创建时间起")
    private LocalDateTime startTime;

    @Schema(description = "创建时间止")
    private LocalDateTime endTime;
}
