package com.pei.dehaze.ui.system.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.sdk.api.DatasetAPI;
import com.pei.dehaze.sdk.model.dataset.Dataset;
import com.pei.dehaze.sdk.model.dataset.DatasetQuery;
import com.pei.dehaze.ui.common.BaseViewModel;

import java.util.Collections;
import java.util.List;

public class DatasetManageViewModel extends BaseViewModel {

    private final MutableLiveData<List<Dataset>> datasetList = new MutableLiveData<>();
    private final MutableLiveData<Long> total = new MutableLiveData<>(0L);

    private int pageNum = 1;
    private int pageSize = 10;
    private String keywords = "";

    public void loadDatasets() {
        DatasetQuery query = new DatasetQuery();
        query.setPageNum(pageNum);
        query.setPageSize(pageSize);
        query.setKeyword(keywords.isEmpty() ? null : keywords);
        DatasetAPI.getList(query, RepositoryAdapters.wrap(withLoading(data -> {
            datasetList.postValue(data.getList());
            total.postValue(data.getTotal());
        })));
    }

    public void addDataset(Dataset data) {
        DatasetAPI.add(data, RepositoryAdapters.wrap(withLoading(v -> {
            operationResult.postValue("新增数据集成功");
            loadDatasets();
        })));
    }

    public void updateDataset(long id, Dataset data) {
        DatasetAPI.update(id, data, RepositoryAdapters.wrap(withLoading(v -> {
            operationResult.postValue("修改数据集成功");
            loadDatasets();
        })));
    }

    public void deleteDataset(long id) {
        DatasetAPI.delete(id, RepositoryAdapters.wrap(withLoading(v -> {
            operationResult.postValue("删除数据集成功");
            loadDatasets();
        })));
    }

    public void setKeywords(String keywords) {
        this.keywords = keywords != null ? keywords : "";
        this.pageNum = 1;
    }

    public void resetQuery() {
        this.keywords = "";
        this.pageNum = 1;
    }

    public void nextPage() {
        long t = total.getValue() != null ? total.getValue() : 0L;
        if (pageNum < (int) Math.ceil(t * 1.0 / pageSize)) {
            pageNum++;
            loadDatasets();
        }
    }

    public void prevPage() {
        if (pageNum > 1) {
            pageNum--;
            loadDatasets();
        }
    }

    public int getPageNum() { return pageNum; }
    public int getPageSize() { return pageSize; }

    public LiveData<List<Dataset>> getDatasetList() { return datasetList; }
    public LiveData<Long> getTotal() { return total; }
}
