package com.pei.dehaze.ui.system.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.sdk.api.AlgorithmAPI;
import com.pei.dehaze.sdk.model.algorithm.Algorithm;
import com.pei.dehaze.sdk.model.algorithm.AlgorithmQuery;
import com.pei.dehaze.sdk.model.algorithm.AlgorithmStatus;
import com.pei.dehaze.ui.common.BaseViewModel;

import java.util.Collections;
import java.util.List;

public class AlgorithmManageViewModel extends BaseViewModel {

    private final MutableLiveData<List<Algorithm>> algorithmList = new MutableLiveData<>();

    private String keywords = "";

    public void loadAlgorithms() {
        AlgorithmQuery query = new AlgorithmQuery();
        query.setKeywords(keywords.isEmpty() ? null : keywords);
        AlgorithmAPI.getList(query, RepositoryAdapters.wrap(withLoading(data -> {
            algorithmList.postValue(data != null ? data : Collections.emptyList());
        })));
    }

    public void updateStatus(long id, AlgorithmStatus status) {
        AlgorithmAPI.updateStatus(id, status, RepositoryAdapters.wrap(withLoading(v -> {
            operationResult.postValue("状态更新成功");
            loadAlgorithms();
        })));
    }

    public void deleteAlgorithm(long id) {
        AlgorithmAPI.deleteByIds(Collections.singletonList(id),
                RepositoryAdapters.wrap(withLoading(v -> {
                    operationResult.postValue("删除成功");
                    loadAlgorithms();
                })));
    }

    public void setKeywords(String keywords) {
        this.keywords = keywords != null ? keywords : "";
    }

    public void resetQuery() {
        this.keywords = "";
    }

    public LiveData<List<Algorithm>> getAlgorithmList() {
        return algorithmList;
    }
}
