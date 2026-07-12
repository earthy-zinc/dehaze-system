package com.pei.dehaze.repository;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.api.AlgorithmAPI;
import com.pei.dehaze.sdk.api.FileAPI;
import com.pei.dehaze.sdk.api.ModelAPI;
import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.algorithm.Algorithm;
import com.pei.dehaze.sdk.model.algorithm.AlgorithmQuery;
import com.pei.dehaze.sdk.model.file.FileInfo;
import com.pei.dehaze.sdk.model.prediction.PredParam;
import com.pei.dehaze.sdk.model.prediction.PredResult;
import com.pei.dehaze.sdk.model.prediction.PredictionLogVO;
import com.pei.dehaze.sdk.network.ApiException;

import java.io.File;
import java.util.List;

public class PresentationRepository {

    public interface AlgorithmListCallback {
        void onSuccess(List<Algorithm> algorithms);
        void onError(String errorMessage);
    }

    public interface AlgorithmDetailCallback {
        void onSuccess(Algorithm algorithm);
        void onError(String errorMessage);
    }

    public interface UploadCallback {
        void onSuccess(FileInfo fileInfo);
        void onError(String errorMessage);
    }

    public interface PredictionCallback {
        void onSuccess(PredResult result);
        void onError(String errorMessage);
    }

    public interface OptionsCallback {
        void onSuccess(List<Option> options);
        void onError(String errorMessage);
    }

    public interface PredictionLogListCallback {
        void onSuccess(List<PredictionLogVO> logs);
        void onError(String errorMessage);
    }

    public void getAlgorithmList(AlgorithmQuery query, AlgorithmListCallback callback) {
        AlgorithmAPI.getList(query, new ApiCallback<List<Algorithm>>() {
            @Override
            public void onSuccess(List<Algorithm> data) {
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

    public void getAlgorithmDetail(int id, AlgorithmDetailCallback callback) {
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

    public void listPredictionLogs(int pageNum, int pageSize, PredictionLogListCallback callback) {
        ModelAPI.listPredictionLogs(null, pageNum, pageSize, new ApiCallback<PageResult<PredictionLogVO>>() {
            @Override
            public void onSuccess(PageResult<PredictionLogVO> data) {
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
