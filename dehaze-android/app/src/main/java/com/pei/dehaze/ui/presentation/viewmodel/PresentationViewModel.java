package com.pei.dehaze.ui.presentation.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import com.pei.dehaze.repository.PresentationRepository;
import com.pei.dehaze.sdk.model.algorithm.Algorithm;
import com.pei.dehaze.sdk.model.algorithm.AlgorithmQuery;
import com.pei.dehaze.sdk.model.model.PredParam;
import com.pei.dehaze.sdk.model.model.PredResult;

import java.io.File;
import java.util.List;

public class PresentationViewModel extends ViewModel {

    private final PresentationRepository presentationRepository;

    private final MutableLiveData<List<Algorithm>> algorithmList = new MutableLiveData<>();
    private final MutableLiveData<String> imageUrl = new MutableLiveData<>();
    private final MutableLiveData<PredResult> predictionResult = new MutableLiveData<>();
    private final MutableLiveData<Boolean> loading = new MutableLiveData<>();
    private final MutableLiveData<String> error = new MutableLiveData<>();

    public PresentationViewModel() {
        presentationRepository = new PresentationRepository();
    }

    public void loadAlgorithms(AlgorithmQuery query) {
        loading.setValue(true);
        presentationRepository.getAlgorithmList(query, new PresentationRepository.AlgorithmListCallback() {
            @Override
            public void onSuccess(List<Algorithm> algorithms) {
                algorithmList.postValue(algorithms);
                loading.postValue(false);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void uploadImage(File imageFile, int modelId) {
        loading.setValue(true);
        presentationRepository.uploadImage(imageFile, modelId, new PresentationRepository.UploadCallback() {
            @Override
            public void onSuccess(String url) {
                imageUrl.postValue(url);
                loading.postValue(false);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void getPrediction(PredParam param) {
        loading.setValue(true);
        presentationRepository.getPrediction(param, new PresentationRepository.PredictionCallback() {
            @Override
            public void onSuccess(PredResult result) {
                predictionResult.postValue(result);
                loading.postValue(false);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public LiveData<List<Algorithm>> getAlgorithmList() {
        return algorithmList;
    }

    public LiveData<String> getImageUrl() {
        return imageUrl;
    }

    public LiveData<PredResult> getPredictionResult() {
        return predictionResult;
    }

    public LiveData<Boolean> getLoading() {
        return loading;
    }

    public LiveData<String> getError() {
        return error;
    }
}