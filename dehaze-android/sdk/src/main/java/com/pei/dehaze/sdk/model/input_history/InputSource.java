package com.pei.dehaze.sdk.model.input_history;

/**
 * 图像来源枚举
 * 对齐后端 InputHistory.inputSource：upload/camera/sample
 */
public enum InputSource {
    UPLOAD("upload", "上传"),
    CAMERA("camera", "相机"),
    SAMPLE("sample", "样本");

    private final String value;
    private final String label;

    InputSource(String value, String label) {
        this.value = value;
        this.label = label;
    }

    public String getValue() {
        return value;
    }

    public String getLabel() {
        return label;
    }

    public static InputSource fromValue(String value) {
        if (value == null) return null;
        for (InputSource source : values()) {
            if (source.value.equalsIgnoreCase(value)) {
                return source;
            }
        }
        return null;
    }
}
