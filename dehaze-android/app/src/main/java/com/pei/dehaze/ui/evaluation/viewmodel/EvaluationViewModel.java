package com.pei.dehaze.ui.evaluation.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import com.pei.dehaze.repository.EvaluationRepository;
import com.pei.dehaze.sdk.model.algorithm.Algorithm;
import com.pei.dehaze.sdk.model.model.EvalParam;
import com.pei.dehaze.sdk.model.model.EvalResult;
import com.pei.dehaze.sdk.model.model.PredParam;
import com.pei.dehaze.sdk.model.model.PredResult;

import java.io.File;
import java.util.List;

public class EvaluationViewModel extends ViewModel {

    private final EvaluationRepository evaluationRepository;

    private final MutableLiveData<String> imageUrl = new MutableLiveData<>();
    private final MutableLiveData<PredResult> predictionResult = new MutableLiveData<>();
    private final MutableLiveData<List<EvalResult>> evaluationResults = new MutableLiveData<>();
    private final MutableLiveData<Algorithm> algorithmInfo = new MutableLiveData<>();
    private final MutableLiveData<Boolean> loading = new MutableLiveData<>();
    private final MutableLiveData<String> error = new MutableLiveData<>();

    public EvaluationViewModel() {
        evaluationRepository = new EvaluationRepository();
    }

    public void uploadImage(File imageFile, int modelId) {
        loading.setValue(true);
        evaluationRepository.uploadImage(imageFile, modelId, new EvaluationRepository.UploadCallback() {
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
        evaluationRepository.getPrediction(param, new EvaluationRepository.PredictionCallback() {
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

    public void getEvaluation(EvalParam param) {
        loading.setValue(true);
        evaluationRepository.getEvaluation(param, new EvaluationRepository.EvaluationCallback() {
            @Override
            public void onSuccess(List<EvalResult> results) {
                evaluationResults.postValue(results);
                loading.postValue(false);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void getAlgorithmInfo(int id) {
        loading.setValue(true);
        evaluationRepository.getAlgorithmInfo(id, new EvaluationRepository.AlgorithmCallback() {
            @Override
            public void onSuccess(Algorithm algorithm) {
                algorithmInfo.postValue(algorithm);
                loading.postValue(false);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public LiveData<String> getImageUrl() {
        return imageUrl;
    }

    public LiveData<PredResult> getPredictionResult() {
        return predictionResult;
    }

    public LiveData<List<EvalResult>> getEvaluationResults() {
        return evaluationResults;
    }

    public LiveData<Algorithm> getAlgorithmInfo() {
        return algorithmInfo;
    }

    public LiveData<Boolean> getLoading() {
        return loading;
    }

    public LiveData<String> getError() {
        return error;
    }
}