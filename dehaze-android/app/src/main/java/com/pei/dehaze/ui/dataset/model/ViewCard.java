package com.pei.dehaze.ui.dataset.model;

/**
 * 用于瀑布流展示的图片卡片模型
 */
public class ViewCard {
    private int id;
    private String src;
    private String originSrc;
    private String alt;

    public ViewCard() {
    }

    public ViewCard(int id, String src, String originSrc, String alt) {
        this.id = id;
        this.src = src;
        this.originSrc = originSrc;
        this.alt = alt;
    }

    public int getId() {
        return id;
    }

    public void setId(int id) {
        this.id = id;
    }

    public String getSrc() {
        return src;
    }

    public void setSrc(String src) {
        this.src = src;
    }

    public String getOriginSrc() {
        return originSrc;
    }

    public void setOriginSrc(String originSrc) {
        this.originSrc = originSrc;
    }

    public String getAlt() {
        return alt;
    }

    public void setAlt(String alt) {
        this.alt = alt;
    }
}