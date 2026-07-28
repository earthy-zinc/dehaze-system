package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@EqualsAndHashCode(callSuper = true)
@Schema(description = "评价详情VO")
public class RatingDetailVO extends RatingPageVO {

    @Schema(description = "算法ID")
    private Long algorithmId;
}
