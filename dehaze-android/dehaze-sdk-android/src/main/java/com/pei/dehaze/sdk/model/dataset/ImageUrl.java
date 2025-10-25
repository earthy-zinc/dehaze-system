package com.pei.dehaze.sdk.model.dataset;

import lombok.Data;

/**
 * 图片URL模型类
 */
@Data
public class ImageUrl {
    private int id;
    
    /**
     * 图片类型（有雾图像、无雾图像）
     */
    private String type;
    
    /**
     * 图片URL
     */
    private String url;
    
    /**
     * 高清图片URL
     */
    private String originUrl;
    
    /**
     * 描述
     */
    private String description;
}