package com.pei.dehaze.ui.dehaze;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.repository.RepositoryCallback;
import com.pei.dehaze.sdk.api.AiImageAPI;
import com.pei.dehaze.sdk.api.AiModelAPI;
import com.pei.dehaze.sdk.model.ai_model.AiModelVO;
import com.pei.dehaze.sdk.model.file.FileInfo;
import com.pei.dehaze.sdk.model.image_task.ImageTaskQuery;
import com.pei.dehaze.sdk.model.image_task.ImageTaskResponse;
import com.pei.dehaze.sdk.model.image_task.ImageTaskVO;
import com.pei.dehaze.ui.common.BaseViewModel;

import java.util.ArrayList;
import java.util.List;

public class DehazeViewModel extends BaseViewModel {

    private final MutableLiveData<Integer> currentStep = new MutableLiveData<>(0);
    private final MutableLiveData<FileInfo> uploadedFile = new MutableLiveData<>();
    private final MutableLiveData<List<AiModelVO>> models = new MutableLiveData<>(new ArrayList<>());
    private final MutableLiveData<String> selectedModelId = new MutableLiveData<>();
    private final MutableLiveData<String> selectedModelName = new MutableLiveData<>();
    private final MutableLiveData<Float> strength = new MutableLiveData<>(0.5f);
    private final MutableLiveData<Float> brightness = new MutableLiveData<>(0.5f);
    private final MutableLiveData<Float> contrast = new MutableLiveData<>(0.5f);
    private final MutableLiveData<ImageTaskResponse> predictionResult = new MutableLiveData<>();
    private final MutableLiveData<Boolean> isProcessing = new MutableLiveData<>(false);

    /** 加载启用的图像编辑模型，预选历史最近使用的模型；历史模型已停用或首次进入时回落到列表首项 */
    public void loadModels() {
        AiModelAPI.listEnabledModels("image_edit", null,
                RepositoryAdapters.wrap(withLoading(list -> {
                    List<AiModelVO> available = list != null ? list : new ArrayList<>();
                    models.postValue(available);
                    if (available.isEmpty() || selectedModelId.getValue() != null) return;

                    ImageTaskQuery query = new ImageTaskQuery();
                    query.setPageNum(1);
                    query.setPageSize(1);
                    AiImageAPI.listImageTasks(query, RepositoryAdapters.wrapPage(
                            new RepositoryCallback<List<ImageTaskVO>>() {
                                @Override
                                public void onSuccess(List<ImageTaskVO> tasks) {
                                    String lastUsed = tasks != null && !tasks.isEmpty()
                                            ? tasks.get(0).getModel() : null;
                                    selectDefaultModel(available, lastUsed);
                                }

                                @Override
                                public void onError(String errorMessage) {
                                    selectDefaultModel(available, null);
                                }
                            }));
                })));
    }

    private void selectDefaultModel(List<AiModelVO> available, String lastUsedModel) {
        if (selectedModelId.getValue() != null) return;
        for (AiModelVO model : available) {
            if (model.getModelId() != null && model.getModelId().equals(lastUsedModel)) {
                setSelectedModel(model);
                return;
            }
        }
        setSelectedModel(available.get(0));
    }

    public void setSelectedModel(AiModelVO model) {
        if (model == null) return;
        selectedModelId.postValue(model.getModelId());
        selectedModelName.postValue(model.getDisplayName());
    }

    public void setCurrentStep(int step) {
        currentStep.postValue(step);
    }

    public void setUploadedFile(FileInfo file) {
        uploadedFile.postValue(file);
    }

    public void setStrength(float value) {
        strength.setValue(value);
    }

    public void setBrightness(float value) {
        brightness.setValue(value);
    }

    public void setContrast(float value) {
        contrast.setValue(value);
    }

    public void setPredictionResult(ImageTaskResponse result) {
        predictionResult.postValue(result);
    }

    public void setProcessing(boolean processing) {
        isProcessing.postValue(processing);
    }

    public void reset() {
        currentStep.setValue(0);
        uploadedFile.postValue(null);
        selectedModelId.postValue(null);
        selectedModelName.postValue(null);
        strength.setValue(0.5f);
        brightness.setValue(0.5f);
        contrast.setValue(0.5f);
        predictionResult.postValue(null);
        isProcessing.postValue(false);
    }

    public LiveData<Integer> getCurrentStep() {
        return currentStep;
    }

    public LiveData<FileInfo> getUploadedFile() {
        return uploadedFile;
    }

    public LiveData<List<AiModelVO>> getModels() {
        return models;
    }

    public LiveData<String> getSelectedModelId() {
        return selectedModelId;
    }

    public LiveData<String> getSelectedModelName() {
        return selectedModelName;
    }

    public LiveData<Float> getStrength() {
        return strength;
    }

    public LiveData<Float> getBrightness() {
        return brightness;
    }

    public LiveData<Float> getContrast() {
        return contrast;
    }

    public LiveData<ImageTaskResponse> getPredictionResult() {
        return predictionResult;
    }

    public LiveData<Boolean> getIsProcessing() {
        return isProcessing;
    }
}
