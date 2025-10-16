package com.pei.dehaze.sdk.model.model;

import lombok.Data;

/**
 * 模型评估参数模型类
 */
@Data
public class EvalParam {
    private int modelId;
    private String predUrl;
    private String gtUrl;
}