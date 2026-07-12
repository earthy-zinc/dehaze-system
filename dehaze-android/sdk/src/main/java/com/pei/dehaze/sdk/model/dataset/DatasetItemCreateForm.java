package com.pei.dehaze.sdk.model.dataset;

import lombok.Data;

/**
 * 数据项创建表单（对齐后端 createDatasetItemRequest）
 */
@Data
public class DatasetItemCreateForm {
    private Long datasetId;
    private String name;
}
