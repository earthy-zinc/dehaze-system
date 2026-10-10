package com.pei.dehaze.ui.batch.viewmodel;

import android.net.Uri;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.repository.RepositoryCallback;
import com.pei.dehaze.sdk.api.AiImageAPI;
import com.pei.dehaze.sdk.api.AiModelAPI;
import com.pei.dehaze.sdk.api.FileAPI;
import com.pei.dehaze.sdk.api.MemberAPI;
import com.pei.dehaze.sdk.model.ai_model.AiModelVO;
import com.pei.dehaze.sdk.model.file.FileInfo;
import com.pei.dehaze.sdk.model.image_task.DehazeParams;
import com.pei.dehaze.sdk.model.image_task.ImageEditForm;
import com.pei.dehaze.sdk.model.image_task.ImageTaskResponse;
import com.pei.dehaze.sdk.model.image_task.ImageTaskStatus;
import com.pei.dehaze.sdk.model.member.MemberProfileVO;
import com.pei.dehaze.ui.batch.model.BatchImageItem;
import com.pei.dehaze.ui.common.BaseViewModel;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * 批量处理 ViewModel：逐张上传换取 fileId，再逐张提交图像任务，串行推进避免并发打爆后端
 */
public class BatchViewModel extends BaseViewModel {

    /** 与 taro/uniapp 端一致：取不到会员权益（非会员/未登录）时的批量上限 */
    private static final int DEFAULT_MAX_IMAGES = 5;

    private final MutableLiveData<List<BatchImageItem>> imageItems =
            new MutableLiveData<>(new ArrayList<>());
    private final MutableLiveData<List<AiModelVO>> models = new MutableLiveData<>(new ArrayList<>());
    private final MutableLiveData<Integer> maxImages = new MutableLiveData<>(DEFAULT_MAX_IMAGES);
    private final MutableLiveData<List<BatchImageItem>> resultItems =
            new MutableLiveData<>(new ArrayList<>());
    private final MutableLiveData<Boolean> isProcessing = new MutableLiveData<>(false);
    private final MutableLiveData<String> progressCount = new MutableLiveData<>("");

    private final List<BatchImageItem> pendingItems = new ArrayList<>();
    private AiModelVO selectedModel;
    private DehazeParams params = new DehazeParams();

    public LiveData<List<BatchImageItem>> getImageItems() { return imageItems; }

    public LiveData<List<AiModelVO>> getModels() { return models; }

    public LiveData<Integer> getMaxImages() { return maxImages; }

    public LiveData<List<BatchImageItem>> getResultItems() { return resultItems; }

    public LiveData<Boolean> getIsProcessing() { return isProcessing; }

    public LiveData<String> getProgressCount() { return progressCount; }

    public AiModelVO getSelectedModel() { return selectedModel; }

    public void setSelectedModel(AiModelVO model) { this.selectedModel = model; }

    public void setParams(DehazeParams params) { this.params = params; }

    /** 加载启用的图像编辑模型与当前用户的批量上限 */
    public void loadOptions() {
        AiModelAPI.listEnabledModels("image_edit", null,
                RepositoryAdapters.wrap(withLoading(list -> {
                    List<AiModelVO> available = list != null ? list : new ArrayList<>();
                    models.postValue(available);
                    if (selectedModel == null && !available.isEmpty()) {
                        selectedModel = available.get(0);
                    }
                })));

        MemberAPI.getProfile(RepositoryAdapters.wrap(new RepositoryCallback<MemberProfileVO>() {
            @Override
            public void onSuccess(MemberProfileVO data) {
                Integer limit = data != null && data.getBenefits() != null
                        ? data.getBenefits().getBatchLimit() : null;
                maxImages.postValue(limit != null && limit > 0 ? limit : DEFAULT_MAX_IMAGES);
            }

            @Override
            public void onError(String errorMessage) {
                // 未登录/非会员取不到权益，按默认上限，不作为错误提示
            }
        }));
    }

    public boolean canAddMore() {
        List<BatchImageItem> current = imageItems.getValue();
        int limit = maxImages.getValue() != null ? maxImages.getValue() : DEFAULT_MAX_IMAGES;
        return current != null && current.size() < limit;
    }

    public void addImage(Uri uri, File file) {
        List<BatchImageItem> current = imageItems.getValue();
        if (current == null) current = new ArrayList<>();
        if (!canAddMore()) return;
        BatchImageItem item = new BatchImageItem(current.size(), uri, file);
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

    public void startBatchProcessing() {
        if (pendingItems.isEmpty() || selectedModel == null) return;
        isProcessing.postValue(true);
        List<BatchImageItem> results = new ArrayList<>(pendingItems);
        resultItems.postValue(results);
        processNext(0, results);
    }

    private void processNext(int index, List<BatchImageItem> results) {
        if (index >= results.size()) {
            isProcessing.postValue(false);
            return;
        }
        BatchImageItem item = results.get(index);
        item.setStatus(BatchImageItem.Status.PROCESSING);
        notifyResultUpdate(results);

        if (item.getFile() == null) {
            failItem(item, "无法读取图片文件", results, index);
            return;
        }
        FileAPI.upload(item.getFile(), RepositoryAdapters.wrap(new RepositoryCallback<FileInfo>() {
            @Override
            public void onSuccess(FileInfo fileInfo) {
                submitTask(item, fileInfo, results, index);
            }

            @Override
            public void onError(String errorMessage) {
                failItem(item, "上传失败: " + errorMessage, results, index);
            }
        }));
    }

    private void submitTask(BatchImageItem item, FileInfo fileInfo,
                            List<BatchImageItem> results, int index) {
        ImageEditForm form = new ImageEditForm();
        form.setModel(selectedModel.getModelId());
        form.setFileId(fileInfo.getId());
        form.setParams(params);

        AiImageAPI.submitAndWait(form, RepositoryAdapters.wrap(new RepositoryCallback<ImageTaskResponse>() {
            @Override
            public void onSuccess(ImageTaskResponse result) {
                if (result == null || result.getStatus() == null) {
                    failItem(item, "处理结果异常", results, index);
                    return;
                }
                if (result.getStatus() != ImageTaskStatus.COMPLETED) {
                    failItem(item, result.getErrorMessage() != null
                            ? result.getErrorMessage() : "处理失败", results, index);
                    return;
                }
                item.setStatus(BatchImageItem.Status.COMPLETED);
                item.setResultUrl(result.getResultUrl());
                item.setTime(result.getTime());
                notifyResultUpdate(results);
                processNext(index + 1, results);
            }

            @Override
            public void onError(String errorMessage) {
                failItem(item, errorMessage, results, index);
            }
        }));
    }

    private void failItem(BatchImageItem item, String message,
                          List<BatchImageItem> results, int index) {
        item.setStatus(BatchImageItem.Status.FAILED);
        item.setErrorMessage(message);
        notifyResultUpdate(results);
        processNext(index + 1, results);
    }

    private void notifyResultUpdate(List<BatchImageItem> results) {
        progressCount.postValue(countCompleted(results) + "/" + results.size());
        resultItems.postValue(new ArrayList<>(results));
    }

    private int countCompleted(List<BatchImageItem> results) {
        int count = 0;
        for (BatchImageItem item : results) {
            if (item.getStatus() == BatchImageItem.Status.COMPLETED
                    || item.getStatus() == BatchImageItem.Status.FAILED) {
                count++;
            }
        }
        return count;
    }

    public void retryItem(BatchImageItem item, List<BatchImageItem> results) {
        int idx = -1;
        for (int i = 0; i < results.size(); i++) {
            if (results.get(i).getIndex() == item.getIndex()) {
                idx = i;
                break;
            }
        }
        if (idx >= 0) {
            item.setStatus(BatchImageItem.Status.PENDING);
            item.setErrorMessage(null);
            item.setResultUrl(null);
            item.setTime(null);
            isProcessing.postValue(true);
            processNext(idx, results);
        }
    }
}
