package com.pei.dehaze.ui.dataset.model;

/**
 * 图片类型模型
 */
public class ImageType {
    private int id;
    private String type;
    private boolean enabled;

    public ImageType() {
    }

    public ImageType(int id, String type, boolean enabled) {
        this.id = id;
        this.type = type;
        this.enabled = enabled;
    }

    public int getId() {
        return id;
    }

    public void setId(int id) {
        this.id = id;
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

    public boolean isEnabled() {
        return enabled;
    }

    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
    }
}