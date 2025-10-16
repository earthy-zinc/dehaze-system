package com.pei.dehaze.sdk.model.dataset;

import lombok.Data;

/**
 * 数据集图片文件信息模型类
 */
@Data
public class DatasetImageFileInfo {
    private int id;
    private int datasetItemId;
    private int fileId;
    private String type;
    private String description;
    private String url;
}