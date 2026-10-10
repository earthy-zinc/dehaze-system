package com.pei.dehaze.ui.batch.model;

import android.net.Uri;

import java.io.File;

/**
 * 批量处理单张图片的状态
 */
public class BatchImageItem {
    public enum Status {
        PENDING, PROCESSING, COMPLETED, FAILED
    }

    private final int index;
    private final Uri uri;
    /** 缓存到本地的副本，用于上传换取 fileId */
    private final File file;
    private Status status = Status.PENDING;
    private String resultUrl;
    /** 推理耗时（秒），取自图像任务响应 time */
    private Integer time;
    private String errorMessage;

    public BatchImageItem(int index, Uri uri, File file) {
        this.index = index;
        this.uri = uri;
        this.file = file;
    }

    public int getIndex() {
        return index;
    }

    public Uri getUri() {
        return uri;
    }

    public File getFile() {
        return file;
    }

    public Status getStatus() {
        return status;
    }

    public void setStatus(Status status) {
        this.status = status;
    }

    public String getResultUrl() {
        return resultUrl;
    }

    public void setResultUrl(String resultUrl) {
        this.resultUrl = resultUrl;
    }

    public Integer getTime() {
        return time;
    }

    public void setTime(Integer time) {
        this.time = time;
    }

    public String getErrorMessage() {
        return errorMessage;
    }

    public void setErrorMessage(String errorMessage) {
        this.errorMessage = errorMessage;
    }
}
