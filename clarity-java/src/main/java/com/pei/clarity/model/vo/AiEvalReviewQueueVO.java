package com.pei.clarity.model.vo;

import java.util.List;
import lombok.Data;

/**
 * 复核队列响应
 *
 * @author clarity
 */
@Data
public class AiEvalReviewQueueVO {

    private List<AiEvalReviewItemVO> items;

    private Integer pending;

    private Integer reviewed;
}
