package com.pei.dehaze.ui.algorithm.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.repository.RepositoryCallback;
import com.pei.dehaze.sdk.api.AlgorithmAPI;
import com.pei.dehaze.ui.common.BaseViewModel;
import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.algorithm.Algorithm;
import com.pei.dehaze.sdk.model.algorithm.AlgorithmQuery;
import com.pei.dehaze.sdk.model.algorithm.AlgorithmStatus;

import java.util.ArrayList;
import java.util.List;

public class AlgorithmViewModel extends BaseViewModel {

    private final MutableLiveData<List<Algorithm>> algorithmList = new MutableLiveData<>();
    private final MutableLiveData<Algorithm> algorithmDetail = new MutableLiveData<>();
    private final MutableLiveData<List<Option>> algorithmOptions = new MutableLiveData<>();

    private String keywords = "";

    public void loadAlgorithms() {
        AlgorithmQuery query = new AlgorithmQuery();
        query.setKeywords(keywords);
        AlgorithmAPI.getList(query, RepositoryAdapters.wrap(withLoading(data ->
                algorithmList.postValue(data != null ? data : new ArrayList<>()))));
    }

    public void loadAlgorithmDetail(long id) {
        AlgorithmAPI.getAlgorithmInfoById(id, RepositoryAdapters.wrap(withLoading(algorithmDetail::postValue)));
    }

    public void addAlgorithm(Algorithm data) {
        AlgorithmAPI.add(data, RepositoryAdapters.wrap(withLoading(v -> {
            operationResult.postValue("新增算法成功");
            loadAlgorithms();
        })));
    }

    public void updateAlgorithm(long id, Algorithm data) {
        AlgorithmAPI.update(id, data, RepositoryAdapters.wrap(withLoading(v -> {
            operationResult.postValue("修改算法成功");
            loadAlgorithms();
        })));
    }

    public void deleteAlgorithms(List<Long> ids) {
        AlgorithmAPI.deleteByIds(ids, RepositoryAdapters.wrap(withLoading(v -> {
            operationResult.postValue("删除算法成功");
            loadAlgorithms();
        })));
    }

    public void updateAlgorithmStatus(long id, AlgorithmStatus status) {
        AlgorithmAPI.updateStatus(id, status, RepositoryAdapters.wrap(withLoading(v -> {
            operationResult.postValue("状态更新成功");
            loadAlgorithms();
        })));
    }

    public void loadOptions() {
        AlgorithmAPI.getOption(RepositoryAdapters.wrap(new RepositoryCallback<List<Option>>() {
            @Override
            public void onSuccess(List<Option> data) {
                algorithmOptions.postValue(data);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
            }
        }));
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

    public LiveData<List<Option>> getAlgorithmOptions() {
        return algorithmOptions;
    }
}
