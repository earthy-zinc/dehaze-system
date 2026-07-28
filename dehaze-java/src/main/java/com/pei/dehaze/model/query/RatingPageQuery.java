package com.pei.dehaze.model.query;

import com.pei.dehaze.common.base.BasePageQuery;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;

import java.time.LocalDateTime;
import java.util.List;

@Data
@EqualsAndHashCode(callSuper = true)
@Schema(description = "评价后台分页查询参数")
public class RatingPageQuery extends BasePageQuery {

    @Schema(description = "用户名/昵称关键字")
    private String keywords;

    @Schema(description = "算法ID")
    private Long algorithmId;

    @Schema(description = "评分下限")
    private Integer ratingMin;

    @Schema(description = "评分上限")
    private Integer ratingMax;

    @Schema(description = "是否有文字")
    private Boolean hasComment;

    @Schema(description = "标签筛选")
    private List<String> tags;

    @Schema(description = "起始时间")
    private LocalDateTime startTime;

    @Schema(description = "结束时间")
    private LocalDateTime endTime;
}
