package com.pei.dehaze.repository;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.api.AlgorithmAPI;
import com.pei.dehaze.sdk.api.ModelAPI;
import com.pei.dehaze.sdk.model.algorithm.Algorithm;
import com.pei.dehaze.sdk.model.model.EvalParam;
import com.pei.dehaze.sdk.model.model.EvalResult;
import com.pei.dehaze.sdk.model.model.PredParam;
import com.pei.dehaze.sdk.model.model.PredResult;

import java.io.File;
import java.util.List;

public class EvaluationRepository {

    public interface UploadCallback {
        void onSuccess(String imageUrl);
        void onError(String errorMessage);
    }

    public interface PredictionCallback {
        void onSuccess(PredResult result);
        void onError(String errorMessage);
    }

    public interface EvaluationCallback {
        void onSuccess(List<EvalResult> results);
        void onError(String errorMessage);
    }

    public interface AlgorithmCallback {
        void onSuccess(Algorithm algorithm);
        void onError(String errorMessage);
    }

    public void uploadImage(File imageFile, int modelId, UploadCallback callback) {
        // TODO: 实现图片上传功能
        // 这里需要使用 Retrofit + MultipartBody 上传图片
        callback.onError("未实现图片上传功能");
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

    public void getAlgorithmInfo(int id, AlgorithmCallback callback) {
        AlgorithmAPI.getAlgorithmInfoById(id, new ApiCallback<Algorithm>() {
            @Override
            public void onSuccess(Algorithm data) {
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