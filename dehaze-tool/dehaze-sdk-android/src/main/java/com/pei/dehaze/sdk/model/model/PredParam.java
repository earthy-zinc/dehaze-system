package com.pei.dehaze.sdk.model.model;

import lombok.Data;

/**
 * 模型预测参数模型类
 */
@Data
public class PredParam {
    private int modelId;
    private String url;
    private Object modelParam;
}