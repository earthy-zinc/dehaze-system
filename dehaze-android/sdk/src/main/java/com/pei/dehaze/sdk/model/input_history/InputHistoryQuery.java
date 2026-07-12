package com.pei.dehaze.sdk.model.input_history;

import com.pei.dehaze.sdk.model.PageQuery;

import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * 图像输入历史分页查询参数
 */
@Data
@EqualsAndHashCode(callSuper = true)
public class InputHistoryQuery extends PageQuery {
    /** 图片来源筛选（upload/camera/sample） */
    private String inputSource;
    /** 仅收藏 */
    private Boolean favoriteOnly = false;
    /** 关键词 */
    private String keywords;
}
