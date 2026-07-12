package com.pei.dehaze.sdk.model.dataset;

import lombok.Data;

/**
 * 图片文件信息（对齐后端 ImageFileInfo）
 */
@Data
public class DatasetImageFileInfo {
    private Long id;
    private Long datasetItemId;
    private Long fileId;
    private String type;
    private String description;
    private String url;
}
