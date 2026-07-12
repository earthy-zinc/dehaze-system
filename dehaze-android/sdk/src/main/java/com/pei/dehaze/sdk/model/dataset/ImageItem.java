package com.pei.dehaze.sdk.model.dataset;

import lombok.Data;

import java.util.List;

/**
 * 数据项模型类（对齐后端 ImageItemVO）
 */
@Data
public class ImageItem {
    private Long id;
    private Long datasetId;
    private String name;
    private Integer imageCount;
    private List<ImageUrl> hazyImages;
    private String createTime;
    private String updateTime;
}
