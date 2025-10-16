package com.pei.dehaze.sdk.model.dict;

import lombok.Data;

/**
 * 字典类型分页对象模型类
 */
@Data
public class DictTypePageVO {
    /**
     * 字典类型ID
     */
    private int id;
    
    /**
     * 类型编码
     */
    private String code;
    
    /**
     * 类型名称
     */
    private String name;
    
    /**
     * 状态(1:启用;0:禁用)
     */
    private Integer status;
    
    /**
     * 备注
     */
    private String remark;
}