package com.pei.dehaze.sdk.model.dataset;

import com.pei.dehaze.sdk.model.PageQuery;
import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * 图片项查询参数模型类
 */
@EqualsAndHashCode(callSuper = true)
@Data
public class ImageItemQuery extends PageQuery {
    private String keywords;
}