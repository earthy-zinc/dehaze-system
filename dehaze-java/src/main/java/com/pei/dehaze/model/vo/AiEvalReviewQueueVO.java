package com.pei.dehaze.model.vo;

import lombok.Data;

import java.util.List;

/**
 * 复核队列响应
 *
 * @author dehaze
 */
@Data
public class AiEvalReviewQueueVO {

    private List<AiEvalReviewItemVO> items;

    private Integer pending;

    private Integer reviewed;
}
