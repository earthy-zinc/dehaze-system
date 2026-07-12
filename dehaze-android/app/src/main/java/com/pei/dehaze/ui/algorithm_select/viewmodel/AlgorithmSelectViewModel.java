package com.pei.dehaze.ui.algorithm_select.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import com.pei.dehaze.repository.AlgorithmSelectRepository;
import com.pei.dehaze.sdk.model.algorithm_select.AlgorithmCompareVO;
import com.pei.dehaze.sdk.model.algorithm_select.AlgorithmRecommendVO;
import com.pei.dehaze.sdk.model.algorithm_select.FavoriteToggleResult;
import com.pei.dehaze.sdk.model.algorithm_select.FavoriteVO;

import java.util.ArrayList;
import java.util.List;

public class AlgorithmSelectViewModel extends ViewModel {

    private final AlgorithmSelectRepository repository;

    private final MutableLiveData<List<AlgorithmRecommendVO>> recommendList = new MutableLiveData<>();
    private final MutableLiveData<List<FavoriteVO>> favoriteList = new MutableLiveData<>();
    private final MutableLiveData<List<AlgorithmCompareVO>> compareResult = new MutableLiveData<>();
    private final MutableLiveData<FavoriteToggleResult> favoriteToggleResult = new MutableLiveData<>();
    private final MutableLiveData<Boolean> loading = new MutableLiveData<>(false);
    private final MutableLiveData<String> error = new MutableLiveData<>();
    private final MutableLiveData<String> operationResult = new MutableLiveData<>();

    public AlgorithmSelectViewModel() {
        repository = new AlgorithmSelectRepository();
    }

    public void recommend(String imageUrl, int topN) {
        loading.setValue(true);
        repository.recommend(imageUrl, topN, new AlgorithmSelectRepository.Callback<List<AlgorithmRecommendVO>>() {
            @Override
            public void onSuccess(List<AlgorithmRecommendVO> data) {
                recommendList.postValue(data != null ? data : new ArrayList<>());
                loading.postValue(false);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void loadFavorites() {
        loading.setValue(true);
        repository.listFavorites(new AlgorithmSelectRepository.Callback<List<FavoriteVO>>() {
            @Override
            public void onSuccess(List<FavoriteVO> data) {
                favoriteList.postValue(data != null ? data : new ArrayList<>());
                loading.postValue(false);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void toggleFavorite(long algorithmId) {
        repository.toggleFavorite(algorithmId, new AlgorithmSelectRepository.Callback<FavoriteToggleResult>() {
            @Override
            public void onSuccess(FavoriteToggleResult data) {
                favoriteToggleResult.postValue(data);
                operationResult.postValue(data.isFavorited() ? "已收藏" : "已取消收藏");
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
            }
        });
    }

    public void compare(List<Long> algorithmIds, String imageUrl) {
        loading.setValue(true);
        repository.compare(algorithmIds, imageUrl, new AlgorithmSelectRepository.Callback<List<AlgorithmCompareVO>>() {
            @Override
            public void onSuccess(List<AlgorithmCompareVO> data) {
                compareResult.postValue(data != null ? data : new ArrayList<>());
                loading.postValue(false);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
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

    public LiveData<Boolean> getLoading() {
        return loading;
    }

    public LiveData<String> getError() {
        return error;
    }

    public LiveData<String> getOperationResult() {
        return operationResult;
    }

    public void clearError() {
        error.setValue(null);
    }

    public void clearOperationResult() {
        operationResult.setValue(null);
    }

    public void clearCompareResult() {
        compareResult.setValue(null);
    }

    public void clearFavoriteToggleResult() {
        favoriteToggleResult.setValue(null);
    }
}
