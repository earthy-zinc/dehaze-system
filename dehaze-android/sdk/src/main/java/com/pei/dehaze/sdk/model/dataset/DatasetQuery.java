package com.pei.dehaze.sdk.model.dataset;

import com.pei.dehaze.sdk.model.PageQuery;
import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * 数据集查询参数（对齐后端 DatasetQuery）
 */
@EqualsAndHashCode(callSuper = true)
@Data
public class DatasetQuery extends PageQuery {
    private String keyword;
    private String type;
    private Integer status;
}
