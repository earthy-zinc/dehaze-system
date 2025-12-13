package com.pei.dehaze.common.enums;

import com.pei.dehaze.common.base.IBaseEnum;

/**
 * @author earthy-zinc
 * @since 2025-12-07 22:38:01
 */
public enum DownloadStatusEnum  implements IBaseEnum<String> {
    PROCESSING("processing", "处理中"),
    COMPLETED("completed", "已完成"),
    FAILED("failed", "失败");

    private final String value;

    private final String label;

    DownloadStatusEnum(String value, String label) {
        this.value = value;
        this.label = label;
    }
    @Override
    public String getValue() {
        return value;
    }

    @Override
    public String getLabel() {
        return label;
    }
}
