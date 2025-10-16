package com.pei.dehaze.sdk.model.model;

import lombok.Data;

/**
 * 模型评估结果模型类
 */
@Data
public class EvalResult {
    private int id;
    // 评价指标的名称
    private String label;
    // 评价指标的值
    private String value;
    // 基准值
    private String baseline;
    // 评价指标是越高越好还是越低越好
    private String better;
    // 评价指标的描述
    private String description;
}