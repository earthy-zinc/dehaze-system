package com.pei.dehaze.ui.algorithm.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.pei.dehaze.repository.AlgorithmRepository;
import com.pei.dehaze.repository.RepositoryCallback;
import com.pei.dehaze.ui.common.BaseViewModel;
import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.algorithm.Algorithm;
import com.pei.dehaze.sdk.model.algorithm.AlgorithmFavorite;
import com.pei.dehaze.sdk.model.algorithm.AlgorithmQuery;
import com.pei.dehaze.sdk.model.algorithm.AlgorithmStatus;

import java.util.ArrayList;
import java.util.List;

public class AlgorithmViewModel extends BaseViewModel {

    private final AlgorithmRepository algorithmRepository;

    private final MutableLiveData<List<Algorithm>> algorithmList = new MutableLiveData<>();
    private final MutableLiveData<Algorithm> algorithmDetail = new MutableLiveData<>();
    private final MutableLiveData<List<Algorithm>> compareResult = new MutableLiveData<>();
    private final MutableLiveData<List<AlgorithmFavorite>> favoriteList = new MutableLiveData<>();
    private final MutableLiveData<List<Option>> algorithmOptions = new MutableLiveData<>();

    private String keywords = "";

    public AlgorithmViewModel() {
        algorithmRepository = new AlgorithmRepository();
    }

    public void loadAlgorithms() {
        AlgorithmQuery query = new AlgorithmQuery();
        query.setKeywords(keywords);
        algorithmRepository.getAlgorithms(query, withLoading(data ->
                algorithmList.postValue(data != null ? data : new ArrayList<>())));
    }

    public void loadAlgorithmDetail(long id) {
        algorithmRepository.getAlgorithmDetail(id, withLoading(algorithmDetail::postValue));
    }

    public void addAlgorithm(Algorithm data) {
        algorithmRepository.addAlgorithm(data, withLoading(v -> {
            operationResult.postValue("新增算法成功");
            loadAlgorithms();
        }));
    }

    public void updateAlgorithm(long id, Algorithm data) {
        algorithmRepository.updateAlgorithm(id, data, withLoading(v -> {
            operationResult.postValue("修改算法成功");
            loadAlgorithms();
        }));
    }

    public void deleteAlgorithms(List<Long> ids) {
        algorithmRepository.deleteAlgorithms(ids, withLoading(v -> {
            operationResult.postValue("删除算法成功");
            loadAlgorithms();
        }));
    }

    public void updateAlgorithmStatus(long id, AlgorithmStatus status) {
        algorithmRepository.updateAlgorithmStatus(id, status, withLoading(v -> {
            operationResult.postValue("状态更新成功");
            loadAlgorithms();
        }));
    }

    public void compareAlgorithms(String ids) {
        algorithmRepository.compare(ids, withLoading(data ->
                compareResult.postValue(data != null ? data : new ArrayList<>())));
    }

    public void loadFavorites() {
        algorithmRepository.listFavorites(withLoading(data ->
                favoriteList.postValue(data != null ? data : new ArrayList<>())));
    }

    public void toggleFavorite(long id) {
        algorithmRepository.toggleFavorite(id, new RepositoryCallback<Void>() {
            @Override
            public void onSuccess(Void data) {
                operationResult.postValue("收藏状态已更新");
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
            }
        });
    }

    public void loadOptions() {
        algorithmRepository.getOptions(new RepositoryCallback<List<Option>>() {
            @Override
            public void onSuccess(List<Option> data) {
                algorithmOptions.postValue(data);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
            }
        });
    }

    public void setKeywords(String keywords) {
        this.keywords = keywords == null ? "" : keywords;
    }

    public void resetQuery() {
        this.keywords = "";
    }

    public LiveData<List<Algorithm>> getAlgorithmList() {
        return algorithmList;
    }

    public LiveData<Algorithm> getAlgorithmDetail() {
        return algorithmDetail;
    }

    public LiveData<List<Algorithm>> getCompareResult() {
        return compareResult;
    }

    public LiveData<List<AlgorithmFavorite>> getFavoriteList() {
        return favoriteList;
    }

    public LiveData<List<Option>> getAlgorithmOptions() {
        return algorithmOptions;
    }

    public void clearCompareResult() {
        compareResult.setValue(null);
    }
}
