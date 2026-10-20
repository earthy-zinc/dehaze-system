package com.pei.clarity.model.vo;

import lombok.Data;

/**
 * 模型降级频次行
 *
 * @author clarity
 */
@Data
public class AiDowngradeVO {

    private String modelId;

    private Long count;
}
