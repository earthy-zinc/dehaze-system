package com.pei.dehaze.repository;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.api.AlgorithmAPI;
import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.algorithm.Algorithm;
import com.pei.dehaze.sdk.model.algorithm.AlgorithmFavorite;
import com.pei.dehaze.sdk.model.algorithm.AlgorithmQuery;
import com.pei.dehaze.sdk.network.ApiException;

import java.util.List;

public class AlgorithmRepository {

    public interface Callback<T> {
        void onSuccess(T data);
        void onError(String errorMessage);
    }

    public void getAlgorithms(AlgorithmQuery query, Callback<List<Algorithm>> callback) {
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

    public void getAlgorithmDetail(int id, Callback<Algorithm> callback) {
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

    public void compare(String ids, Callback<List<Algorithm>> callback) {
        AlgorithmAPI.compare(ids, new ApiCallback<List<Algorithm>>() {
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

    public void getOptions(Callback<List<Option>> callback) {
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

    public void listFavorites(Callback<List<AlgorithmFavorite>> callback) {
        AlgorithmAPI.listFavorites(new ApiCallback<List<AlgorithmFavorite>>() {
            @Override
            public void onSuccess(List<AlgorithmFavorite> data) {
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

    public void toggleFavorite(long id, Callback<Void> callback) {
        AlgorithmAPI.toggleFavorite(id, new ApiCallback<Void>() {
            @Override
            public void onSuccess(Void data) {
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

    public void addAlgorithm(Algorithm data, Callback<Void> callback) {
        AlgorithmAPI.add(data, new ApiCallback<Void>() {
            @Override
            public void onSuccess(Void data) {
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

    public void updateAlgorithm(int id, Algorithm data, Callback<Void> callback) {
        AlgorithmAPI.update(id, data, new ApiCallback<Void>() {
            @Override
            public void onSuccess(Void data) {
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

    public void updateAlgorithmStatus(long id, int status, Callback<Void> callback) {
        AlgorithmAPI.updateStatus(id, status, new ApiCallback<Void>() {
            @Override
            public void onSuccess(Void data) {
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

    public void deleteAlgorithms(String ids, Callback<Void> callback) {
        AlgorithmAPI.deleteByIds(ids, new ApiCallback<Void>() {
            @Override
            public void onSuccess(Void data) {
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
}
