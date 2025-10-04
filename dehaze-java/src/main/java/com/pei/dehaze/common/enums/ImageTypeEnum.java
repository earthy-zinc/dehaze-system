package com.pei.dehaze.common.enums;

import com.pei.dehaze.common.base.IBaseEnum;
import lombok.Getter;

/**
 * @author earthy-zinc
 * @since 2024-06-08 23:29:05
 */
@Getter
public enum ImageTypeEnum implements IBaseEnum<String> {
    UPLOAD("upload", "上传图片"),
    DATASET("dataset", "数据集图片"),
    PREDICT("predict", "预测图片"),

    HAZE("有雾图像", "有雾图像"),
    PRED("预测图像", "预测图像"),
    CLEAN("清晰图像", "清晰图像");

    private final String value;

    private final String label;

    ImageTypeEnum(String value, String label) {
        this.value = value;
        this.label = label;
    }
}
