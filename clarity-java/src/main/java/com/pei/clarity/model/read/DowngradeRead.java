package com.pei.clarity.model.read;

import lombok.Data;

/**
 * 模型降级次数行
 *
 * @author clarity
 */
@Data
public class DowngradeRead {

    private String modelId;

    private Long cnt;
}
