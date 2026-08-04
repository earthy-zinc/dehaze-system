package com.pei.dehaze.sdk.model.algorithm_select;

import lombok.Data;

/**
 * 自定义图片测试表单
 * 对齐后端 AlgorithmTestForm（/api/v1/algorithms/select/{id}/test）
 */
@Data
public class AlgorithmTestForm {
    /** 文件ID（与imageUrl二选一） */
    private Long fileId;
    /** 图片URL（与fileId二选一） */
    private String imageUrl;
    /** 预测参数（JSON） */
    private String params;
}
