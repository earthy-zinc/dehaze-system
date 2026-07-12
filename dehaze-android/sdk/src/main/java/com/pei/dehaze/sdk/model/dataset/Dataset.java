package com.pei.dehaze.sdk.model.dataset;

import lombok.Data;

import java.util.List;

/**
 * 数据集模型类（对齐后端 DatasetVO）
 */
@Data
public class Dataset {
    private Long id;
    private Long parentId;
    private String type;
    private String name;
    private String description;
    private String path;
    private String size;
    private Boolean hasChildren;
    private List<Dataset> children;
    private Integer status;
    private DatasetStatistics statistics;
    private Long total;
    private String createTime;
    private String updateTime;
}
