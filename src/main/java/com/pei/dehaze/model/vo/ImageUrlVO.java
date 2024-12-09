package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

/**
 * @author earthy-zinc
 * @since 2024-06-08 18:47:10
 */
@Data
public class ImageUrlVO {
    @Schema(description = "ItemFileId")
    private Long id;

    private String type;

    private String url;

    private String originUrl;

    private String description;
}
