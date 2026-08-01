package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.util.List;

@Data
@Schema(description = "算法选择树节点")
public class AlgorithmSelectNodeVO {

    @Schema(description = "节点ID")
    private Long id;

    @Schema(description = "父节点ID")
    private Long parentId;

    @Schema(description = "节点名称")
    private String name;

    @Schema(description = "算法类型")
    private String type;

    @Schema(description = "是否为叶子节点（算法节点）")
    private boolean leaf;

    @Schema(description = "子节点")
    private List<AlgorithmSelectNodeVO> children;
}
