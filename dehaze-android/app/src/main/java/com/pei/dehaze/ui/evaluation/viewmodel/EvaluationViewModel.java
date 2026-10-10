package com.pei.dehaze.ui.evaluation.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.sdk.api.AiImageAPI;
import com.pei.dehaze.sdk.api.AiModelAPI;
import com.pei.dehaze.sdk.api.FileAPI;
import com.pei.dehaze.sdk.api.ModelAPI;
import com.pei.dehaze.ui.common.BaseViewModel;
import com.pei.dehaze.sdk.model.ai_model.AiModelVO;
import com.pei.dehaze.sdk.model.evaluation.EvalParam;
import com.pei.dehaze.sdk.model.evaluation.EvalResult;
import com.pei.dehaze.sdk.model.evaluation.EvalTaskStatus;
import com.pei.dehaze.sdk.model.evaluation.EvaluationLogVO;
import com.pei.dehaze.sdk.model.file.FileInfo;
import com.pei.dehaze.sdk.model.image_task.DehazeParams;
import com.pei.dehaze.sdk.model.image_task.ImageEditForm;
import com.pei.dehaze.sdk.model.image_task.ImageTaskResponse;
import com.pei.dehaze.sdk.model.image_task.ImageTaskStatus;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class EvaluationViewModel extends BaseViewModel {

    private final MutableLiveData<FileInfo> hazyFile = new MutableLiveData<>();
    private final MutableLiveData<FileInfo> clearFile = new MutableLiveData<>();
    private final MutableLiveData<List<AiModelVO>> models = new MutableLiveData<>();
    private final MutableLiveData<ImageTaskResponse> predictionResult = new MutableLiveData<>();
    private final MutableLiveData<EvalResult> evaluationResult = new MutableLiveData<>();
    private final MutableLiveData<List<EvaluationLogVO>> evaluationLogs = new MutableLiveData<>();

    public void uploadHazyImage(File imageFile) {
        FileAPI.upload(imageFile, RepositoryAdapters.wrap(withLoading(fileInfo -> {
            hazyFile.postValue(fileInfo);
            operationResult.postValue("有雾图上传成功");
        })));
    }

    public void uploadClearImage(File imageFile) {
        FileAPI.upload(imageFile, RepositoryAdapters.wrap(withLoading(fileInfo -> {
            clearFile.postValue(fileInfo);
            operationResult.postValue("清晰图上传成功");
        })));
    }

    public void loadModels() {
        AiModelAPI.listEnabledModels("image_edit", null,
                RepositoryAdapters.wrap(withLoading(models::postValue)));
    }

    public void predict(String modelId, DehazeParams params) {
        FileInfo hazy = hazyFile.getValue();
        if (hazy == null || hazy.getUrl() == null) {
            error.setValue("请先上传有雾图片");
            return;
        }
        DehazeParams actualParams = params != null ? params : new DehazeParams();
        String invalidMsg = actualParams.validate();
        if (invalidMsg != null) {
            error.setValue(invalidMsg);
            return;
        }
        ImageEditForm form = new ImageEditForm();
        form.setModel(modelId);
        form.setImageUrl(hazy.getUrl());
        form.setParams(actualParams);
        AiImageAPI.submitAndWait(form, RepositoryAdapters.wrap(withLoading(result -> {
            if (result.getStatus() == ImageTaskStatus.FAILED) {
                error.postValue(result.getErrorMessage() != null ? result.getErrorMessage() : "去雾处理失败");
                return;
            }
            predictionResult.postValue(result);
            operationResult.postValue("去雾处理完成，可进行评估");
        })));
    }

    public void evaluate(String modelId) {
        ImageTaskResponse pred = predictionResult.getValue();
        FileInfo clear = clearFile.getValue();
        if (pred == null || pred.getResultUrl() == null) {
            error.setValue("请先执行去雾处理");
            return;
        }
        if (clear == null || clear.getUrl() == null) {
            error.setValue("请先上传清晰参考图");
            return;
        }
        EvalParam param = new EvalParam();
        param.setModelId(modelId);
        param.setPredUrl(pred.getResultUrl());
        param.setGtUrl(clear.getUrl());
        ModelAPI.evaluateAndWait(param, RepositoryAdapters.wrap(withLoading(result -> {
            if (result.getStatus() == EvalTaskStatus.FAILED) {
                error.postValue(result.getErrorMessage() != null ? result.getErrorMessage() : "评估失败");
                return;
            }
            evaluationResult.postValue(result);
            operationResult.postValue("评估完成");
            loadEvaluationLogs();
        })));
    }

    public void loadEvaluationLogs() {
        ModelAPI.listEvaluationLogs(null, 1, 20, RepositoryAdapters.wrapPage(withLoading(logs ->
                evaluationLogs.postValue(logs != null ? logs : new ArrayList<>()))));
    }

    public LiveData<FileInfo> getHazyFile() {
        return hazyFile;
    }

    public LiveData<FileInfo> getClearFile() {
        return clearFile;
    }

    public LiveData<List<AiModelVO>> getModels() {
        return models;
    }

    public LiveData<ImageTaskResponse> getPredictionResult() {
        return predictionResult;
    }

    public LiveData<EvalResult> getEvaluationResult() {
        return evaluationResult;
    }

    public LiveData<List<EvaluationLogVO>> getEvaluationLogs() {
        return evaluationLogs;
    }
}
