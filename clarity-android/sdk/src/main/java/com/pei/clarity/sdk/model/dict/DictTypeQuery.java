package com.pei.clarity.sdk.model.dict;

import com.pei.clarity.sdk.model.PageQuery;
import lombok.Data;

/**
 * 字典类型查询参数模型类
 */
@Data
public class DictTypeQuery extends PageQuery {
    private String keywords;
}