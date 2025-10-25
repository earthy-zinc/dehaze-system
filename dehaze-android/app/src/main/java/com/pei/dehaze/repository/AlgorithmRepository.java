package com.pei.dehaze.repository;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.api.AlgorithmAPI;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.algorithm.Algorithm;
import com.pei.dehaze.sdk.model.algorithm.AlgorithmQuery;

import java.util.List;

public class AlgorithmRepository {

    public interface AlgorithmCallback {
        void onSuccess(List<Algorithm> algorithms);
        void onError(String errorMessage);
    }

    public interface AlgorithmDetailCallback {
        void onSuccess(Algorithm algorithm);
        void onError(String errorMessage);
    }

    public void getAlgorithms(AlgorithmQuery query, AlgorithmCallback callback) {
        AlgorithmAPI.getList(query, new ApiCallback<List<Algorithm>>() {
            @Override
            public void onSuccess(List<Algorithm> data) {
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

    public void getAlgorithmDetail(int id, AlgorithmDetailCallback callback) {
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