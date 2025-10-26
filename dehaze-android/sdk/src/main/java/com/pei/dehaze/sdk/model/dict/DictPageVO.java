package com.pei.dehaze.sdk.model.dict;

import lombok.Data;

/**
 * 字典分页对象模型类
 */
@Data
public class DictPageVO {
    /**
     * 字典ID
     */
    private Integer id;
    
    /**
     * 字典名称
     */
    private String name;
    
    /**
     * 状态(1:启用;0:禁用)
     */
    private Integer status;
    
    /**
     * 字典值
     */
    private String value;
}