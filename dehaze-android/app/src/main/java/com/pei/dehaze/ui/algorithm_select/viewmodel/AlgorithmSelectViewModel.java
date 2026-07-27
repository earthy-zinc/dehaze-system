package com.pei.dehaze.ui.algorithm_select.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.repository.RepositoryCallback;
import com.pei.dehaze.sdk.api.AlgorithmSelectAPI;
import com.pei.dehaze.ui.common.BaseViewModel;
import com.pei.dehaze.sdk.model.algorithm_select.AlgorithmCompareVO;
import com.pei.dehaze.sdk.model.algorithm_select.AlgorithmRecommendVO;
import com.pei.dehaze.sdk.model.algorithm_select.FavoriteToggleResult;
import com.pei.dehaze.sdk.model.algorithm_select.FavoriteVO;

import java.util.ArrayList;
import java.util.List;

public class AlgorithmSelectViewModel extends BaseViewModel {

    private final MutableLiveData<List<AlgorithmRecommendVO>> recommendList = new MutableLiveData<>();
    private final MutableLiveData<List<FavoriteVO>> favoriteList = new MutableLiveData<>();
    private final MutableLiveData<List<AlgorithmCompareVO>> compareResult = new MutableLiveData<>();
    private final MutableLiveData<FavoriteToggleResult> favoriteToggleResult = new MutableLiveData<>();

    public void recommend(String imageUrl, int topN) {
        AlgorithmSelectAPI.recommend(imageUrl, topN, RepositoryAdapters.wrap(withLoading(data ->
                recommendList.postValue(data != null ? data : new ArrayList<>()))));
    }

    public void loadFavorites() {
        AlgorithmSelectAPI.listFavorites(RepositoryAdapters.wrap(withLoading(data ->
                favoriteList.postValue(data != null ? data : new ArrayList<>()))));
    }

    public void toggleFavorite(long algorithmId) {
        AlgorithmSelectAPI.toggleFavorite(algorithmId, RepositoryAdapters.wrap(new RepositoryCallback<FavoriteToggleResult>() {
            @Override
            public void onSuccess(FavoriteToggleResult data) {
                favoriteToggleResult.postValue(data);
                operationResult.postValue(data.isFavorited() ? "已收藏" : "已取消收藏");
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
            }
        }));
    }

    public void compare(List<Long> algorithmIds, String imageUrl) {
        AlgorithmSelectAPI.compare(algorithmIds, imageUrl, RepositoryAdapters.wrap(withLoading(data ->
                compareResult.postValue(data != null ? data : new ArrayList<>()))));
    }

    public LiveData<List<AlgorithmRecommendVO>> getRecommendList() {
        return recommendList;
    }

    public LiveData<List<FavoriteVO>> getFavoriteList() {
        return favoriteList;
    }

    public LiveData<List<AlgorithmCompareVO>> getCompareResult() {
        return compareResult;
    }

    public LiveData<FavoriteToggleResult> getFavoriteToggleResult() {
        return favoriteToggleResult;
    }

    public void clearCompareResult() {
        compareResult.setValue(null);
    }

    public void clearFavoriteToggleResult() {
        favoriteToggleResult.setValue(null);
    }
}
