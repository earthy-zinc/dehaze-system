package com.pei.dehaze.model.vo;

import lombok.Data;

/**
 * 人工复核统计
 *
 * @author dehaze
 */
@Data
public class AiJudgeReviewStatsVO {

    private Integer total;

    private Integer pending;

    private Integer reviewed;

    private Integer agreeCount;

    private Integer disagreeCount;

    private Double agreementRate;
}
