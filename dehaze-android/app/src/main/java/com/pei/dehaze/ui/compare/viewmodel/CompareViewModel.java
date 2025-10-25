package com.pei.dehaze.ui.compare.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import com.pei.dehaze.repository.CompareRepository;
import com.pei.dehaze.sdk.model.model.EvalParam;
import com.pei.dehaze.sdk.model.model.EvalResult;
import com.pei.dehaze.sdk.model.model.PredParam;
import com.pei.dehaze.sdk.model.model.PredResult;

import java.util.List;

public class CompareViewModel extends ViewModel {

    private final CompareRepository compareRepository;

    private final MutableLiveData<PredResult> predictionResult = new MutableLiveData<>();
    private final MutableLiveData<List<EvalResult>> evaluationResults = new MutableLiveData<>();
    private final MutableLiveData<Boolean> loading = new MutableLiveData<>();
    private final MutableLiveData<String> error = new MutableLiveData<>();

    public CompareViewModel() {
        compareRepository = new CompareRepository();
    }

    public void loadPrediction(PredParam param) {
        loading.setValue(true);
        compareRepository.getPrediction(param, new CompareRepository.PredictionCallback() {
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

    public void loadEvaluation(EvalParam param) {
        loading.setValue(true);
        compareRepository.getEvaluation(param, new CompareRepository.EvaluationCallback() {
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

    public LiveData<PredResult> getPredictionResult() {
        return predictionResult;
    }

    public LiveData<List<EvalResult>> getEvaluationResults() {
        return evaluationResults;
    }

    public LiveData<Boolean> getLoading() {
        return loading;
    }

    public LiveData<String> getError() {
        return error;
    }
}