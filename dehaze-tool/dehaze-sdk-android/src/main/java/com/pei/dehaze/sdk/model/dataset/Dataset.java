package com.pei.dehaze.sdk.model.dataset;

import lombok.Data;
import java.util.Date;
import java.util.List;

/**
 * 数据集模型类
 */
@Data
public class Dataset {
    /**
     * 数据集ID
     */
    private int id;
    
    /**
     * 父数据集ID
     */
    private int parentId;
    
    /**
     * 数据集类型
     */
    private String type;
    
    /**
     * 数据集名称
     */
    private String name;
    
    /**
     * 数据集描述
     */
    private String description;
    
    /**
     * 存储位置
     */
    private String path;
    
    /**
     * 占用空间大小
     */
    private String size;
    
    /**
     * 数据项数量（简单理解为图片数量）
     */
    private int total;
    
    /**
     * 子数据集
     */
    private List<Dataset> children;
    
    /**
     * 创建时间
     */
    private Date createTime;
    
    /**
     * 修改时间
     */
    private Date updateTime;
    
    /**
     * 状态(1:启用；0:禁用)
     */
    private Integer status;
}