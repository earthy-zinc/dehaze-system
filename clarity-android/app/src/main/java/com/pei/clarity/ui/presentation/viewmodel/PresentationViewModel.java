package com.pei.clarity.ui.presentation.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.pei.clarity.repository.RepositoryAdapters;
import com.pei.clarity.sdk.api.AiImageAPI;
import com.pei.clarity.sdk.api.AiModelAPI;
import com.pei.clarity.sdk.api.FileAPI;
import com.pei.clarity.ui.common.BaseViewModel;
import com.pei.clarity.sdk.model.ai_model.AiModelVO;
import com.pei.clarity.sdk.model.file.FileInfo;
import com.pei.clarity.sdk.model.image_task.DehazeParams;
import com.pei.clarity.sdk.model.image_task.ImageEditForm;
import com.pei.clarity.sdk.model.image_task.ImageTaskQuery;
import com.pei.clarity.sdk.model.image_task.ImageTaskResponse;
import com.pei.clarity.sdk.model.image_task.ImageTaskStatus;
import com.pei.clarity.sdk.model.image_task.ImageTaskVO;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class PresentationViewModel extends BaseViewModel {

    private final MutableLiveData<FileInfo> uploadedFile = new MutableLiveData<>();
    private final MutableLiveData<List<AiModelVO>> models = new MutableLiveData<>();
    private final MutableLiveData<ImageTaskResponse> predictionResult = new MutableLiveData<>();
    private final MutableLiveData<List<ImageTaskVO>> historyList = new MutableLiveData<>();

    private String originalImageUrl;

    public void uploadImage(File imageFile) {
        FileAPI.upload(imageFile, RepositoryAdapters.wrap(withLoading(fileInfo -> {
            uploadedFile.postValue(fileInfo);
            originalImageUrl = fileInfo.getUrl();
            operationResult.postValue("图片上传成功");
        })));
    }

    public void loadModels() {
        AiModelAPI.listEnabledModels("image_edit", null,
                RepositoryAdapters.wrap(withLoading(models::postValue)));
    }

    public void predict(String modelId, DehazeParams params) {
        if (originalImageUrl == null) {
            error.setValue("请先上传图片");
            return;
        }
        String invalidMsg = params == null ? null : params.validate();
        if (invalidMsg != null) {
            error.setValue(invalidMsg);
            return;
        }
        ImageEditForm form = new ImageEditForm();
        form.setModel(modelId);
        form.setImageUrl(originalImageUrl);
        form.setParams(params);
        AiImageAPI.submitAndWait(form, RepositoryAdapters.wrap(withLoading(result -> {
            if (result.getStatus() == ImageTaskStatus.FAILED) {
                error.postValue(result.getErrorMessage() != null ? result.getErrorMessage() : "去雾处理失败");
                return;
            }
            predictionResult.postValue(result);
            operationResult.postValue("去雾处理完成");
            loadHistory();
        })));
    }

    public void loadHistory() {
        ImageTaskQuery query = new ImageTaskQuery();
        query.setPageNum(1);
        query.setPageSize(20);
        AiImageAPI.listImageTasks(query, RepositoryAdapters.wrapPage(withLoading(logs ->
                historyList.postValue(logs != null ? logs : new ArrayList<>()))));
    }

    public LiveData<FileInfo> getUploadedFile() {
        return uploadedFile;
    }

    public LiveData<List<AiModelVO>> getModels() {
        return models;
    }

    public LiveData<ImageTaskResponse> getPredictionResult() {
        return predictionResult;
    }

    public LiveData<List<ImageTaskVO>> getHistoryList() {
        return historyList;
    }

    public String getOriginalImageUrl() {
        return originalImageUrl;
    }
}
