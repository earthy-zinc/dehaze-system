package com.pei.dehaze.repository;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.api.AlgorithmSelectAPI;
import com.pei.dehaze.sdk.model.algorithm_select.AlgorithmCompareVO;
import com.pei.dehaze.sdk.model.algorithm_select.AlgorithmRecommendVO;
import com.pei.dehaze.sdk.model.algorithm_select.FavoriteToggleResult;
import com.pei.dehaze.sdk.model.algorithm_select.FavoriteVO;
import com.pei.dehaze.sdk.network.ApiException;

import java.util.List;

public class AlgorithmSelectRepository {

    public interface Callback<T> {
        void onSuccess(T data);
        void onError(String errorMessage);
    }

    public void recommend(String imageUrl, int topN, Callback<List<AlgorithmRecommendVO>> callback) {
        AlgorithmSelectAPI.recommend(imageUrl, topN, new ApiCallback<List<AlgorithmRecommendVO>>() {
            @Override
            public void onSuccess(List<AlgorithmRecommendVO> data) {
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

    public void toggleFavorite(long algorithmId, Callback<FavoriteToggleResult> callback) {
        AlgorithmSelectAPI.toggleFavorite(algorithmId, new ApiCallback<FavoriteToggleResult>() {
            @Override
            public void onSuccess(FavoriteToggleResult data) {
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

    public void listFavorites(Callback<List<FavoriteVO>> callback) {
        AlgorithmSelectAPI.listFavorites(new ApiCallback<List<FavoriteVO>>() {
            @Override
            public void onSuccess(List<FavoriteVO> data) {
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

    public void compare(List<Long> algorithmIds, String imageUrl, Callback<List<AlgorithmCompareVO>> callback) {
        AlgorithmSelectAPI.compare(algorithmIds, imageUrl, new ApiCallback<List<AlgorithmCompareVO>>() {
            @Override
            public void onSuccess(List<AlgorithmCompareVO> data) {
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
