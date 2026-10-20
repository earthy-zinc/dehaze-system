package com.pei.clarity.sdk.model.dict;

import com.pei.clarity.sdk.model.PageQuery;
import lombok.Data;

/**
 * 字典查询参数模型类
 */
@Data
public class DictQuery extends PageQuery {
    private String name;
    private String typeCode;
}