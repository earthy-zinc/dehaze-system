package com.pei.dehaze.ui.batch.model;

import android.net.Uri;

/**
 * 批量处理单张图片的状态
 */
public class BatchImageItem {
    public enum Status {
        PENDING, PROCESSING, COMPLETED, FAILED
    }

    private final int index;
    private final Uri uri;
    private Status status = Status.PENDING;
    private String resultUrl;
    private String errorMessage;

    public BatchImageItem(int index, Uri uri) {
        this.index = index;
        this.uri = uri;
    }

    public int getIndex() {
        return index;
    }

    public Uri getUri() {
        return uri;
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

    public String getErrorMessage() {
        return errorMessage;
    }

    public void setErrorMessage(String errorMessage) {
        this.errorMessage = errorMessage;
    }
}
