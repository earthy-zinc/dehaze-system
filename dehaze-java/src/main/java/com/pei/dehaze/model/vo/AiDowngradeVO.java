package com.pei.dehaze.model.vo;

import lombok.Data;

/**
 * 模型降级频次行
 *
 * @author dehaze
 */
@Data
public class AiDowngradeVO {

    private String modelId;

    private Long count;
}
