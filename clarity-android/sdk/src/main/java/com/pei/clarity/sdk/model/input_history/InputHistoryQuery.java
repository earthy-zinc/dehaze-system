package com.pei.clarity.sdk.model.input_history;

import com.pei.clarity.sdk.model.PageQuery;

import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * 图像输入历史分页查询参数
 */
@Data
@EqualsAndHashCode(callSuper = true)
public class InputHistoryQuery extends PageQuery {
    /** 图片来源筛选 */
    private InputSource inputSource;
    /** 处理状态筛选（1=成功，2=失败，3=处理中） */
    private Integer status;
    /** 关键词 */
    private String keywords;
}
