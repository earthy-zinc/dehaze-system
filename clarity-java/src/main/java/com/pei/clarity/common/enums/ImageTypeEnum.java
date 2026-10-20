package com.pei.clarity.common.enums;

import com.pei.clarity.common.base.IBaseEnum;
import lombok.Getter;

/**
 * 数据项图片类型枚举
 *
 * @author earthy-zinc
 * @since 2024-06-08 23:29:05
 */
@Getter
public enum ImageTypeEnum implements IBaseEnum<String> {
    HAZE("有雾图像", "有雾图像"),
    CLEAN("清晰图像", "清晰图像");

    private final String value;

    private final String label;

    ImageTypeEnum(String value, String label) {
        this.value = value;
        this.label = label;
    }
}
