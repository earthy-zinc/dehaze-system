package com.pei.dehaze.model.read;

import lombok.Data;

/**
 * 模型降级次数行
 *
 * @author dehaze
 */
@Data
public class DowngradeRead {

    private String modelId;

    private Long cnt;
}
