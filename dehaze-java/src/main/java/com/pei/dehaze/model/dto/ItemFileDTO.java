package com.pei.dehaze.model.dto;

import lombok.Data;
import lombok.EqualsAndHashCode;

@EqualsAndHashCode(callSuper = true)
@Data
public class ItemFileDTO extends FileDTO {
    /**
     * 图片类型
     */
    private String type;

    /**
     * 图片描述
     */
    private String description;

    /**
     * 图片宽度
     */
    private Integer width;

    /**
     * 图片高度
     */
    private Integer height;

    /**
     * 场景类型
     */
    private String sceneType;

    /**
     * 雾霾程度
     */
    private String hazeLevel;
}
