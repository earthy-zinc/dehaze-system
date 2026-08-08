package com.pei.dehaze.ui.dehaze;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.pei.dehaze.sdk.model.algorithm.Algorithm;
import com.pei.dehaze.sdk.model.file.FileInfo;
import com.pei.dehaze.sdk.model.prediction.PredResult;
import com.pei.dehaze.ui.common.BaseViewModel;

import java.util.ArrayList;
import java.util.List;

public class DehazeViewModel extends BaseViewModel {

    private final MutableLiveData<Integer> currentStep = new MutableLiveData<>(0);
    private final MutableLiveData<FileInfo> uploadedFile = new MutableLiveData<>();
    private final MutableLiveData<Long> selectedAlgorithmId = new MutableLiveData<>();
    private final MutableLiveData<String> selectedAlgorithmName = new MutableLiveData<>();
    private final MutableLiveData<Float> strength = new MutableLiveData<>(0.5f);
    private final MutableLiveData<Float> brightness = new MutableLiveData<>(0.5f);
    private final MutableLiveData<Float> contrast = new MutableLiveData<>(0.5f);
    private final MutableLiveData<PredResult> predictionResult = new MutableLiveData<>();
    private final MutableLiveData<Boolean> isProcessing = new MutableLiveData<>(false);

    public void setCurrentStep(int step) {
        currentStep.setValue(step);
    }

    public void setUploadedFile(FileInfo file) {
        uploadedFile.setValue(file);
    }

    public void setSelectedAlgorithm(long id, String name) {
        selectedAlgorithmId.setValue(id);
        selectedAlgorithmName.setValue(name);
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

    public void setPredictionResult(PredResult result) {
        predictionResult.setValue(result);
    }

    public void setProcessing(boolean processing) {
        isProcessing.setValue(processing);
    }

    public void reset() {
        currentStep.setValue(0);
        uploadedFile.setValue(null);
        selectedAlgorithmId.setValue(null);
        selectedAlgorithmName.setValue(null);
        strength.setValue(0.5f);
        brightness.setValue(0.5f);
        contrast.setValue(0.5f);
        predictionResult.setValue(null);
        isProcessing.setValue(false);
    }

    public LiveData<Integer> getCurrentStep() {
        return currentStep;
    }

    public LiveData<FileInfo> getUploadedFile() {
        return uploadedFile;
    }

    public LiveData<Long> getSelectedAlgorithmId() {
        return selectedAlgorithmId;
    }

    public LiveData<String> getSelectedAlgorithmName() {
        return selectedAlgorithmName;
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

    public LiveData<PredResult> getPredictionResult() {
        return predictionResult;
    }

    public LiveData<Boolean> getIsProcessing() {
        return isProcessing;
    }
}
