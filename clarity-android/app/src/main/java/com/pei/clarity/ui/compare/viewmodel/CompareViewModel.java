package com.pei.clarity.ui.compare.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.pei.clarity.repository.RepositoryAdapters;
import com.pei.clarity.repository.RepositoryCallback;
import com.pei.clarity.sdk.api.AiImageAPI;
import com.pei.clarity.sdk.api.AiModelAPI;
import com.pei.clarity.sdk.api.FileAPI;
import com.pei.clarity.sdk.api.ModelAPI;
import com.pei.clarity.ui.common.BaseViewModel;
import com.pei.clarity.sdk.model.ai_model.AiModelVO;
import com.pei.clarity.sdk.model.evaluation.EvalParam;
import com.pei.clarity.sdk.model.evaluation.EvalResult;
import com.pei.clarity.sdk.model.evaluation.EvalTaskStatus;
import com.pei.clarity.sdk.model.file.FileInfo;
import com.pei.clarity.sdk.model.image_task.DehazeParams;
import com.pei.clarity.sdk.model.image_task.ImageEditForm;
import com.pei.clarity.sdk.model.image_task.ImageTaskQuery;
import com.pei.clarity.sdk.model.image_task.ImageTaskResponse;
import com.pei.clarity.sdk.model.image_task.ImageTaskStatus;
import com.pei.clarity.sdk.model.image_task.ImageTaskVO;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

public class CompareViewModel extends BaseViewModel {

    private final MutableLiveData<FileInfo> uploadedFile = new MutableLiveData<>();
    private final MutableLiveData<FileInfo> uploadedGtFile = new MutableLiveData<>();
    private final MutableLiveData<List<AiModelVO>> models = new MutableLiveData<>();
    private final MutableLiveData<ImageTaskResponse> predictionResult = new MutableLiveData<>();
    private final MutableLiveData<Map<String, ImageTaskResponse>> multiPredictionResults = new MutableLiveData<>();
    private final MutableLiveData<EvalResult> evaluationResult = new MutableLiveData<>();
    private final MutableLiveData<List<ImageTaskVO>> predictionLogs = new MutableLiveData<>();

    private String originalImageUrl;
    private String currentModelId;

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
        currentModelId = modelId;
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
        })));
    }

    public void predictMultiple(List<String> modelIds, DehazeParams params) {
        if (originalImageUrl == null) {
            error.setValue("请先上传图片");
            return;
        }
        if (modelIds == null || modelIds.isEmpty()) {
            error.setValue("请至少选择一个模型");
            return;
        }
        String invalidMsg = params == null ? null : params.validate();
        if (invalidMsg != null) {
            error.setValue(invalidMsg);
            return;
        }
        loading.setValue(true);
        currentModelId = modelIds.get(0);
        Map<String, ImageTaskResponse> results = new HashMap<>();
        AtomicInteger pending = new AtomicInteger(modelIds.size());
        for (String modelId : modelIds) {
            ImageEditForm form = new ImageEditForm();
            form.setModel(modelId);
            form.setImageUrl(originalImageUrl);
            form.setParams(params);
            AiImageAPI.submitAndWait(form, RepositoryAdapters.wrap(new RepositoryCallback<ImageTaskResponse>() {
                @Override
                public void onSuccess(ImageTaskResponse result) {
                    if (result.getStatus() == ImageTaskStatus.FAILED) {
                        error.postValue(result.getErrorMessage() != null ? result.getErrorMessage() : "去雾处理失败");
                    } else {
                        synchronized (results) {
                            results.put(modelId, result);
                        }
                    }
                    if (pending.decrementAndGet() == 0) {
                        multiPredictionResults.postValue(results);
                        if (!results.isEmpty()) {
                            operationResult.postValue("多模型处理完成");
                        }
                        loading.postValue(false);
                    }
                }

                @Override
                public void onError(String errorMessage) {
                    if (pending.decrementAndGet() == 0) {
                        multiPredictionResults.postValue(results);
                        loading.postValue(false);
                    }
                    error.postValue(errorMessage);
                }
            }));
        }
    }

    public void evaluate(String modelId, String predUrl, String gtUrl) {
        if (gtUrl == null || gtUrl.isEmpty()) {
            error.setValue("评估需要提供参考图片");
            return;
        }
        EvalParam param = new EvalParam();
        param.setModelId(modelId);
        param.setPredUrl(predUrl);
        param.setGtUrl(gtUrl);
        ModelAPI.evaluateAndWait(param, RepositoryAdapters.wrap(withLoading(result -> {
            if (result.getStatus() == EvalTaskStatus.FAILED) {
                error.postValue(result.getErrorMessage() != null ? result.getErrorMessage() : "评估失败");
                return;
            }
            evaluationResult.postValue(result);
            operationResult.postValue("评估完成");
        })));
    }

    public void loadPredictionLogs() {
        ImageTaskQuery query = new ImageTaskQuery();
        query.setPageNum(1);
        query.setPageSize(20);
        AiImageAPI.listImageTasks(query, RepositoryAdapters.wrapPage(withLoading(logs ->
                predictionLogs.postValue(logs != null ? logs : new ArrayList<>()))));
    }

    /**
     * 上传参考图（GT）用于指标评估。
     */
    public void uploadGtImage(File gtFile) {
        FileAPI.upload(gtFile, RepositoryAdapters.wrap(withLoading(fileInfo -> {
            uploadedGtFile.postValue(fileInfo);
            operationResult.postValue("参考图上传成功");
        })));
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

    public LiveData<Map<String, ImageTaskResponse>> getMultiPredictionResults() {
        return multiPredictionResults;
    }

    public LiveData<EvalResult> getEvaluationResult() {
        return evaluationResult;
    }

    public LiveData<List<ImageTaskVO>> getPredictionLogs() {
        return predictionLogs;
    }

    public String getOriginalImageUrl() {
        return originalImageUrl;
    }

    public LiveData<FileInfo> getUploadedGtFile() {
        return uploadedGtFile;
    }

    public String getCurrentModelId() {
        return currentModelId;
    }
}
