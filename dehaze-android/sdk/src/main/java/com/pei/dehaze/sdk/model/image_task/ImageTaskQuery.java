package com.pei.dehaze.sdk.model.image_task;

import com.pei.dehaze.sdk.model.PageQuery;

import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * 图像任务历史查询参数
 */
@Data
@EqualsAndHashCode(callSuper = true)
public class ImageTaskQuery extends PageQuery {
    /** 按图像模型标识筛选 */
    private String model;
}
