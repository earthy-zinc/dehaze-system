package com.pei.dehaze.sdk.model.dataset;

import lombok.Data;

/**
 * 图片信息更新表单（对齐后端 ItemFileUpdateForm）
 */
@Data
public class ItemFileUpdateForm {
    private String type;
    private String sceneType;
    private String hazeLevel;
    private String description;
}
