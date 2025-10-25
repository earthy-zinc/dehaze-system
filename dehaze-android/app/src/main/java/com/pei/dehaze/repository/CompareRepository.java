package com.pei.dehaze.repository;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.api.ModelAPI;
import com.pei.dehaze.sdk.model.model.EvalParam;
import com.pei.dehaze.sdk.model.model.EvalResult;
import com.pei.dehaze.sdk.model.model.PredParam;
import com.pei.dehaze.sdk.model.model.PredResult;

import java.util.List;

public class CompareRepository {

    public interface PredictionCallback {
        void onSuccess(PredResult result);
        void onError(String errorMessage);
    }

    public interface EvaluationCallback {
        void onSuccess(List<EvalResult> results);
        void onError(String errorMessage);
    }

    public void getPrediction(PredParam param, PredictionCallback callback) {
        ModelAPI.prediction(param, new ApiCallback<PredResult>() {
            @Override
            public void onSuccess(PredResult data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(int code, String message) {
                callback.onError("Error " + code + ": " + message);
            }

            @Override
            public void onFailure(com.pei.dehaze.sdk.network.ApiException e) {
                callback.onError("Network error: " + e.getMessage());
            }
        });
    }

    public void getEvaluation(EvalParam param, EvaluationCallback callback) {
        ModelAPI.evaluation(param, new ApiCallback<List<EvalResult>>() {
            @Override
            public void onSuccess(List<EvalResult> data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(int code, String message) {
                callback.onError("Error " + code + ": " + message);
            }

            @Override
            public void onFailure(com.pei.dehaze.sdk.network.ApiException e) {
                callback.onError("Network error: " + e.getMessage());
            }
        });
    }
}