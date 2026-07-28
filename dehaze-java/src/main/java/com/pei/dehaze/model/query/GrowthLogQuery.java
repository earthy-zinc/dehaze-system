package com.pei.dehaze.model.query;

import com.pei.dehaze.common.base.BasePageQuery;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;

import java.time.LocalDate;

@Data
@EqualsAndHashCode(callSuper = true)
@Schema(description = "成长值流水分页查询对象")
public class GrowthLogQuery extends BasePageQuery {

    @Schema(description = "变动类型(dehaze/evaluate/rating/sign_in/sign_in_bonus/consume/refund_deduct/admin_adjust)")
    private String changeType;

    @Schema(description = "起始时间")
    private LocalDate startTime;

    @Schema(description = "结束时间")
    private LocalDate endTime;

    @Schema(description = "用户ID（后台查询时使用，用户端忽略）")
    private Long userId;
}
