package com.pei.dehaze.sdk.model.dataset;

import lombok.Getter;

@Getter
public enum ImageType {
    CLEAR("clear"),
    HAZY("hazy"),
    TRANS("trans");

    private final String value;

    ImageType(String value) {
        this.value = value;
    }

    public static ImageType fromValue(String value) {
        if (value == null) return null;
        for (ImageType type : values()) {
            if (type.value.equals(value)) return type;
        }
        return null;
    }
}
