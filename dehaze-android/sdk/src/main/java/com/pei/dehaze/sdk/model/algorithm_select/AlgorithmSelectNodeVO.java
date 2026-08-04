package com.pei.dehaze.sdk.model.algorithm_select;

import lombok.Data;

import java.util.List;

/**
 * 算法选择树节点
 * 对齐后端 AlgorithmSelectNodeVO（/api/v1/algorithms/select/tree、search）
 */
@Data
public class AlgorithmSelectNodeVO {
    /** 算法ID */
    private Long id;
    /** 父节点ID（根节点为0） */
    private Long parentId;
    private String name;
    /** 算法类型 */
    private String type;
    /** 是否为叶子节点（算法节点） */
    private Boolean leaf;
    private List<AlgorithmSelectNodeVO> children;
}
