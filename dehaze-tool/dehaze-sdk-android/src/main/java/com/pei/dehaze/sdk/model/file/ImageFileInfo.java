package com.pei.dehaze.sdk.model.file;

import lombok.Data;

/**
 * 图片文件信息模型类
 */
@Data
public class ImageFileInfo {
    /**
     * 所属数据集id
     */
    private Integer datasetId;
    
    /**
     * 所属文件id
     */
    private int fileId;
    
    /**
     * 当前图片id
     */
    private Integer id;
    
    /**
     * 所属数据项id
     */
    private Integer imageItemId;
    
    /**
     * 文件名称
     */
    private String name;
    
    /**
     * 图片类型
     */
    private String type;
    
    /**
     * 文件URL
     */
    private String url;
}