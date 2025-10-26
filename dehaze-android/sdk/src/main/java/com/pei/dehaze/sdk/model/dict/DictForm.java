package com.pei.dehaze.sdk.model.dict;

import lombok.Data;

/**
 * 字典表单模型类
 */
@Data
public class DictForm {
    /**
     * 字典ID
     */
    private Integer id;
    
    /**
     * 字典名称
     */
    private String name;
    
    /**
     * 排序
     */
    private Integer sort;
    
    /**
     * 状态(1:启用;0:禁用)
     */
    private Integer status;
    
    /**
     * 类型编码
     */
    private String typeCode;
    
    /**
     * 值
     */
    private String value;
    
    /**
     * 备注
     */
    private String remark;
}