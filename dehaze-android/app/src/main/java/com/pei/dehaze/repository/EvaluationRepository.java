package com.pei.dehaze.repository;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.api.AlgorithmAPI;
import com.pei.dehaze.sdk.api.FileAPI;
import com.pei.dehaze.sdk.api.ModelAPI;
import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.algorithm.Algorithm;
import com.pei.dehaze.sdk.model.evaluation.EvalParam;
import com.pei.dehaze.sdk.model.evaluation.EvalResult;
import com.pei.dehaze.sdk.model.evaluation.EvaluationLogVO;
import com.pei.dehaze.sdk.model.file.FileInfo;
import com.pei.dehaze.sdk.model.prediction.PredParam;
import com.pei.dehaze.sdk.model.prediction.PredResult;
import com.pei.dehaze.sdk.network.ApiException;

import java.io.File;
import java.util.List;

public class EvaluationRepository {

    public interface UploadCallback {
        void onSuccess(FileInfo fileInfo);
        void onError(String errorMessage);
    }

    public interface PredictionCallback {
        void onSuccess(PredResult result);
        void onError(String errorMessage);
    }

    public interface EvaluationCallback {
        void onSuccess(EvalResult result);
        void onError(String errorMessage);
    }

    public interface AlgorithmCallback {
        void onSuccess(Algorithm algorithm);
        void onError(String errorMessage);
    }

    public interface OptionsCallback {
        void onSuccess(List<Option> options);
        void onError(String errorMessage);
    }

    public interface EvaluationLogListCallback {
        void onSuccess(List<EvaluationLogVO> logs);
        void onError(String errorMessage);
    }

    public void uploadImage(File imageFile, UploadCallback callback) {
        FileAPI.upload(imageFile, new ApiCallback<FileInfo>() {
            @Override
            public void onSuccess(FileInfo data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }

    public void getAlgorithmOptions(OptionsCallback callback) {
        AlgorithmAPI.getOption(new ApiCallback<List<Option>>() {
            @Override
            public void onSuccess(List<Option> data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }

    public void getPrediction(PredParam param, PredictionCallback callback) {
        ModelAPI.predict(param, new ApiCallback<PredResult>() {
            @Override
            public void onSuccess(PredResult data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }

    public void getEvaluation(EvalParam param, EvaluationCallback callback) {
        ModelAPI.evaluate(param, new ApiCallback<EvalResult>() {
            @Override
            public void onSuccess(EvalResult data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
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
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }

    public void listEvaluationLogs(int pageNum, int pageSize, EvaluationLogListCallback callback) {
        ModelAPI.listEvaluationLogs(null, pageNum, pageSize, new ApiCallback<PageResult<EvaluationLogVO>>() {
            @Override
            public void onSuccess(PageResult<EvaluationLogVO> data) {
                callback.onSuccess(data.getList());
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }
}
