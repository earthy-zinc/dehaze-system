package com.pei.dehaze.sdk.model.dataset;

import lombok.Data;

/**
 * 图片URL模型类（对齐后端 ImageUrlVO）
 */
@Data
public class ImageUrl {
    private Long id;
    private String type;
    private String url;
    private String originUrl;
    private String description;
}
