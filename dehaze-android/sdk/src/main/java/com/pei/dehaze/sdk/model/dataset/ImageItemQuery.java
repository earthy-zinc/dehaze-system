package com.pei.dehaze.sdk.model.dataset;

import com.pei.dehaze.sdk.model.PageQuery;
import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * 数据项查询参数
 */
@EqualsAndHashCode(callSuper = true)
@Data
public class ImageItemQuery extends PageQuery {
    private Long datasetId;
    private String keyword;
    private String sceneType;
    private String hazeLevel;
}
