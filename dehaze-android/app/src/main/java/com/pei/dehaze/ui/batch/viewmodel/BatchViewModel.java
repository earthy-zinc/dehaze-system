package com.pei.dehaze.ui.batch.viewmodel;

import android.net.Uri;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.repository.RepositoryCallback;
import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.api.AlgorithmAPI;
import com.pei.dehaze.sdk.api.FileAPI;
import com.pei.dehaze.sdk.api.ModelAPI;
import com.pei.dehaze.sdk.model.algorithm.Algorithm;
import com.pei.dehaze.sdk.model.algorithm.AlgorithmQuery;
import com.pei.dehaze.sdk.model.file.FileInfo;
import com.pei.dehaze.sdk.model.prediction.PredParam;
import com.pei.dehaze.sdk.model.prediction.PredResult;
import com.pei.dehaze.ui.batch.model.BatchImageItem;
import com.pei.dehaze.ui.common.BaseViewModel;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * 批量处理 ViewModel
 */
public class BatchViewModel extends BaseViewModel {

    private static final int MAX_IMAGES = 20;

    private final MutableLiveData<List<BatchImageItem>> imageItems = new MutableLiveData<>(new ArrayList<>());
    private final MutableLiveData<List<Algorithm>> algorithms = new MutableLiveData<>();
    private final MutableLiveData<List<BatchImageItem>> resultItems = new MutableLiveData<>(new ArrayList<>());
    private final MutableLiveData<Boolean> isProcessing = new MutableLiveData<>(false);
    private final MutableLiveData<String> progressCount = new MutableLiveData<>("");

    private final List<BatchImageItem> pendingItems = new ArrayList<>();
    private long selectedAlgorithmId = -1;

    public LiveData<List<BatchImageItem>> getImageItems() { return imageItems; }
    public LiveData<List<Algorithm>> getAlgorithms() { return algorithms; }
    public LiveData<List<BatchImageItem>> getResultItems() { return resultItems; }
    public LiveData<Boolean> getIsProcessing() { return isProcessing; }
    public LiveData<String> getProgressCount() { return progressCount; }
    public long getSelectedAlgorithmId() { return selectedAlgorithmId; }
    public void setSelectedAlgorithmId(long id) { this.selectedAlgorithmId = id; }

    public void addImage(Uri uri) {
        List<BatchImageItem> current = imageItems.getValue();
        if (current == null) current = new ArrayList<>();
        if (current.size() >= MAX_IMAGES) return;
        BatchImageItem item = new BatchImageItem(current.size(), uri);
        current.add(item);
        imageItems.postValue(current);
        pendingItems.add(item);
    }

    public void removeImage(int index) {
        List<BatchImageItem> current = imageItems.getValue();
        if (current != null && index >= 0 && index < current.size()) {
            current.remove(index);
            pendingItems.clear();
            for (BatchImageItem it : current) pendingItems.add(it);
            imageItems.postValue(current);
        }
    }

    public void loadAlgorithms() {
        AlgorithmQuery query = new AlgorithmQuery();
        AlgorithmAPI.getList(query, RepositoryAdapters.wrap(new RepositoryCallback<List<Algorithm>>() {
            @Override
            public void onSuccess(List<Algorithm> data) {
                loading.setValue(false);
                if (data != null) algorithms.postValue(data);
            }
            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.setValue(false);
            }
        }));
    }

    public void startBatchProcessing() {
        if (pendingItems.isEmpty()) return;
        if (selectedAlgorithmId <= 0) return;
        isProcessing.postValue(true);
        List<BatchImageItem> currentResults = new ArrayList<>(pendingItems);
        resultItems.postValue(currentResults);
        processNext(0, currentResults);
    }

    private void processNext(int index, List<BatchImageItem> results) {
        if (index >= results.size()) { isProcessing.postValue(false); return; }
        BatchImageItem item = results.get(index);
        item.setStatus(BatchImageItem.Status.PROCESSING);
        notifyResultUpdate(results);
        File file = getFileFromUri(item.getUri());
        if (file == null) {
            item.setStatus(BatchImageItem.Status.FAILED);
            item.setErrorMessage("无法读取图片文件");
            notifyResultUpdate(results);
            processNext(index + 1, results);
            return;
        }
        FileAPI.upload(file, new ApiCallback<FileInfo>() {
            @Override
            public void onSuccess(FileInfo fileInfo) {
                PredParam param = new PredParam();
                param.setAlgorithmId(selectedAlgorithmId);
                param.setImageUrl(fileInfo.getUrl());
                ModelAPI.predictAndWait(param, new ApiCallback<PredResult>() {
                    @Override
                    public void onSuccess(PredResult predResult) {
                        if (predResult.getResultUrl() != null) {
                            item.setStatus(BatchImageItem.Status.COMPLETED);
                            item.setResultUrl(predResult.getResultUrl());
                        } else {
                            item.setStatus(BatchImageItem.Status.FAILED);
                            item.setErrorMessage(predResult.getErrorMessage() != null ? predResult.getErrorMessage() : "处理失败");
                        }
                        notifyResultUpdate(results);
                        processNext(index + 1, results);
                    }
                    @Override public void onError(String code, String message) {
                        item.setStatus(BatchImageItem.Status.FAILED);
                        item.setErrorMessage(message);
                        notifyResultUpdate(results);
                        processNext(index + 1, results);
                    }
                    @Override public void onFailure(com.pei.dehaze.sdk.network.ApiException e) {
                        item.setStatus(BatchImageItem.Status.FAILED);
                        item.setErrorMessage(e.getMessage());
                        notifyResultUpdate(results);
                        processNext(index + 1, results);
                    }
                });
            }
            @Override public void onError(String code, String message) {
                item.setStatus(BatchImageItem.Status.FAILED);
                item.setErrorMessage("上传失败: " + message);
                notifyResultUpdate(results);
                processNext(index + 1, results);
            }
            @Override public void onFailure(com.pei.dehaze.sdk.network.ApiException e) {
                item.setStatus(BatchImageItem.Status.FAILED);
                item.setErrorMessage("上传失败: " + e.getMessage());
                notifyResultUpdate(results);
                processNext(index + 1, results);
            }
        });
    }

    private void notifyResultUpdate(List<BatchImageItem> results) {
        progressCount.postValue(countCompleted(results) + "/" + results.size());
        resultItems.postValue(new ArrayList<>(results));
    }

    private int countCompleted(List<BatchImageItem> results) {
        int count = 0;
        for (BatchImageItem item : results) {
            if (item.getStatus() == BatchImageItem.Status.COMPLETED || item.getStatus() == BatchImageItem.Status.FAILED) count++;
        }
        return count;
    }

    public void retryItem(BatchImageItem item, List<BatchImageItem> results) {
        int idx = -1;
        for (int i = 0; i < results.size(); i++) {
            if (results.get(i).getIndex() == item.getIndex()) { idx = i; break; }
        }
        if (idx >= 0) {
            item.setStatus(BatchImageItem.Status.PENDING);
            item.setErrorMessage(null);
            item.setResultUrl(null);
            isProcessing.postValue(true);
            processNext(idx, results);
        }
    }

    private File getFileFromUri(Uri uri) {
        if (uri.getScheme() != null && uri.getScheme().equals("file")) return new File(uri.getPath());
        return null;
    }
}
