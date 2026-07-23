package com.pei.dehaze.repository;

import com.pei.dehaze.sdk.api.AlgorithmSelectAPI;
import com.pei.dehaze.sdk.model.algorithm_select.AlgorithmCompareVO;
import com.pei.dehaze.sdk.model.algorithm_select.AlgorithmRecommendVO;
import com.pei.dehaze.sdk.model.algorithm_select.FavoriteToggleResult;
import com.pei.dehaze.sdk.model.algorithm_select.FavoriteVO;

import java.util.List;

public class AlgorithmSelectRepository {

    public void recommend(String imageUrl, int topN, RepositoryCallback<List<AlgorithmRecommendVO>> callback) {
        AlgorithmSelectAPI.recommend(imageUrl, topN, RepositoryAdapters.wrap(callback));
    }

    public void toggleFavorite(long algorithmId, RepositoryCallback<FavoriteToggleResult> callback) {
        AlgorithmSelectAPI.toggleFavorite(algorithmId, RepositoryAdapters.wrap(callback));
    }

    public void listFavorites(RepositoryCallback<List<FavoriteVO>> callback) {
        AlgorithmSelectAPI.listFavorites(RepositoryAdapters.wrap(callback));
    }

    public void compare(List<Long> algorithmIds, String imageUrl, RepositoryCallback<List<AlgorithmCompareVO>> callback) {
        AlgorithmSelectAPI.compare(algorithmIds, imageUrl, RepositoryAdapters.wrap(callback));
    }
}
