package com.pei.dehaze.ui.algorithm_select.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.repository.RepositoryCallback;
import com.pei.dehaze.sdk.api.AlgorithmSelectAPI;
import com.pei.dehaze.sdk.api.FavoriteAPI;
import com.pei.dehaze.sdk.api.RecommendationAPI;
import com.pei.dehaze.sdk.model.algorithm_select.AlgorithmCompareForm;
import com.pei.dehaze.sdk.model.algorithm_select.AlgorithmCompareVO;
import com.pei.dehaze.sdk.model.favorite.FavoriteForm;
import com.pei.dehaze.sdk.model.favorite.FavoriteQuery;
import com.pei.dehaze.sdk.model.favorite.FavoriteVO;
import com.pei.dehaze.sdk.model.recommendation.AnalyzeForm;
import com.pei.dehaze.sdk.model.recommendation.ImageFeatureAnalysisVO;
import com.pei.dehaze.sdk.model.recommendation.RecommendedAlgorithmVO;
import com.pei.dehaze.ui.common.BaseViewModel;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class AlgorithmSelectViewModel extends BaseViewModel {

    private final MutableLiveData<List<RecommendedAlgorithmVO>> recommendList = new MutableLiveData<>();
    private final MutableLiveData<List<FavoriteVO>> favoriteList = new MutableLiveData<>();
    private final MutableLiveData<List<AlgorithmCompareVO>> compareResult = new MutableLiveData<>();

    /**
     * 智能推荐：先做图像特征分析，再基于 imageMd5 获取推荐算法并截取 topN
     */
    public void recommend(String imageUrl, int topN) {
        loading.setValue(true);
        AnalyzeForm form = new AnalyzeForm();
        form.setImageUrl(imageUrl);
        RecommendationAPI.analyze(form, RepositoryAdapters.wrap(new RepositoryCallback<ImageFeatureAnalysisVO>() {
            @Override
            public void onSuccess(ImageFeatureAnalysisVO analysis) {
                RecommendationAPI.getAlgorithmRecommendations(null, analysis.getImageMd5(),
                        RepositoryAdapters.wrap(new RepositoryCallback<List<RecommendedAlgorithmVO>>() {
                            @Override
                            public void onSuccess(List<RecommendedAlgorithmVO> data) {
                                List<RecommendedAlgorithmVO> list = data != null ? data : new ArrayList<>();
                                if (topN > 0 && list.size() > topN) {
                                    list = new ArrayList<>(list.subList(0, topN));
                                }
                                recommendList.postValue(list);
                                loading.postValue(false);
                            }

                            @Override
                            public void onError(String errorMessage) {
                                error.postValue(errorMessage);
                                loading.postValue(false);
                            }
                        }));
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        }));
    }

    public void loadFavorites() {
        FavoriteAPI.getPage(new FavoriteQuery(), RepositoryAdapters.wrapPage(withLoading(list ->
                favoriteList.postValue(list != null ? list : new ArrayList<>()))));
    }

    public void addFavorite(long algorithmId) {
        FavoriteForm form = new FavoriteForm();
        form.setTargetType("algorithm");
        form.setTargetId(algorithmId);
        FavoriteAPI.add(form, RepositoryAdapters.wrap(withLoading(id ->
                operationResult.postValue("已收藏"))));
    }

    public void removeFavorite(long favoriteId) {
        FavoriteAPI.deleteByIds(Collections.singletonList(favoriteId), RepositoryAdapters.wrap(withLoading(v ->
                operationResult.postValue("已取消收藏"))));
    }

    public void compare(List<Long> algorithmIds, String imageUrl) {
        AlgorithmCompareForm form = new AlgorithmCompareForm();
        form.setAlgorithmIds(algorithmIds);
        form.setImageUrl(imageUrl);
        AlgorithmSelectAPI.compare(form, RepositoryAdapters.wrap(withLoading(data ->
                compareResult.postValue(data != null ? data : new ArrayList<>()))));
    }

    public LiveData<List<RecommendedAlgorithmVO>> getRecommendList() {
        return recommendList;
    }

    public LiveData<List<FavoriteVO>> getFavoriteList() {
        return favoriteList;
    }

    public LiveData<List<AlgorithmCompareVO>> getCompareResult() {
        return compareResult;
    }

    public void clearCompareResult() {
        compareResult.setValue(null);
    }
}
