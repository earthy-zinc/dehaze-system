package com.pei.dehaze.sdk.model.dict;

import lombok.Data;

/**
 * 字典表单模型类
 */
@Data
public class DictTypeForm {
    /**
     * 字典类型ID
     */
    private Integer id;
    
    /**
     * 类型名称
     */
    private String name;
    
    /**
     * 类型编码
     */
    private String code;
    
    /**
     * 类型状态：1:启用;0:禁用
     */
    private int status;
    
    /**
     * 备注
     */
    private String remark;
}