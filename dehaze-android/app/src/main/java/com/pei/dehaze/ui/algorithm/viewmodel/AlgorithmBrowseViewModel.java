package com.pei.dehaze.ui.algorithm.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.repository.RepositoryCallback;
import com.pei.dehaze.sdk.api.AlgorithmSelectAPI;
import com.pei.dehaze.sdk.api.RecommendationAPI;
import com.pei.dehaze.sdk.model.algorithm_select.AlgorithmDetailVO;
import com.pei.dehaze.sdk.model.algorithm_select.AlgorithmSelectNodeVO;
import com.pei.dehaze.sdk.model.recommendation.RecommendedAlgorithmVO;
import com.pei.dehaze.ui.common.BaseViewModel;

import java.util.ArrayList;
import java.util.List;

/**
 * 算法库浏览版 ViewModel
 */
public class AlgorithmBrowseViewModel extends BaseViewModel {

    private final MutableLiveData<List<AlgorithmSelectNodeVO>> algorithmList = new MutableLiveData<>();
    private final MutableLiveData<AlgorithmDetailVO> algorithmDetail = new MutableLiveData<>();
    private final MutableLiveData<List<RecommendedAlgorithmVO>> recommendations = new MutableLiveData<>();

    public LiveData<List<AlgorithmSelectNodeVO>> getAlgorithmList() { return algorithmList; }
    public LiveData<AlgorithmDetailVO> getAlgorithmDetail() { return algorithmDetail; }
    public LiveData<List<RecommendedAlgorithmVO>> getRecommendations() { return recommendations; }

    public void loadAlgorithmTree() {
        loading.setValue(true);
        AlgorithmSelectAPI.getTree(RepositoryAdapters.wrap(new RepositoryCallback<List<AlgorithmSelectNodeVO>>() {
            @Override
            public void onSuccess(List<AlgorithmSelectNodeVO> data) {
                loading.setValue(false);
                algorithmList.postValue(data != null ? data : new ArrayList<>());
            }
            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.setValue(false);
            }
        }));
    }

    public void search(String keyword) {
        loading.setValue(true);
        AlgorithmSelectAPI.search(keyword, RepositoryAdapters.wrap(new RepositoryCallback<List<AlgorithmSelectNodeVO>>() {
            @Override
            public void onSuccess(List<AlgorithmSelectNodeVO> data) {
                loading.setValue(false);
                algorithmList.postValue(data != null ? data : new ArrayList<>());
            }
            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.setValue(false);
            }
        }));
    }

    public void loadAlgorithmDetail(long id) {
        loading.setValue(true);
        AlgorithmSelectAPI.getDetail(id, RepositoryAdapters.wrap(new RepositoryCallback<AlgorithmDetailVO>() {
            @Override
            public void onSuccess(AlgorithmDetailVO data) {
                loading.setValue(false);
                algorithmDetail.postValue(data);
            }
            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.setValue(false);
            }
        }));
    }

    public void loadRecommendations(Long analysisId, String imageMd5) {
        loading.setValue(true);
        RecommendationAPI.getAlgorithmRecommendations(analysisId, imageMd5,
                RepositoryAdapters.wrap(new RepositoryCallback<List<RecommendedAlgorithmVO>>() {
                    @Override
                    public void onSuccess(List<RecommendedAlgorithmVO> data) {
                        loading.setValue(false);
                        recommendations.postValue(data != null ? data : new ArrayList<>());
                    }
                    @Override
                    public void onError(String errorMessage) {
                        error.postValue(errorMessage);
                        loading.setValue(false);
                    }
                }));
    }
}
