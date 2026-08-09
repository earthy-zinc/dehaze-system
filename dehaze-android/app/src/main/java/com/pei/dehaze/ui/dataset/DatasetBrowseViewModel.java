package com.pei.dehaze.ui.dataset;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.repository.RepositoryCallback;
import com.pei.dehaze.sdk.api.DatasetAPI;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.dataset.Dataset;
import com.pei.dehaze.sdk.model.dataset.DatasetQuery;
import com.pei.dehaze.ui.common.BaseViewModel;

import java.util.ArrayList;
import java.util.List;

/**
 * 数据集浏览版 ViewModel（仅公开/共享浏览，无管理操作）
 */
public class DatasetBrowseViewModel extends BaseViewModel {

    private final MutableLiveData<List<Dataset>> datasetList = new MutableLiveData<>();

    public LiveData<List<Dataset>> getDatasetList() {
        return datasetList;
    }

    public void loadTree() {
        DatasetAPI.getTree(RepositoryAdapters.wrap(new RepositoryCallback<List<Dataset>>() {
            @Override
            public void onSuccess(List<Dataset> data) {
                loading.setValue(false);
                datasetList.postValue(data != null ? data : new ArrayList<>());
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.setValue(false);
            }
        }));
    }

    public void search(String keywords) {
        loading.setValue(true);
        DatasetQuery query = new DatasetQuery();
        query.setKeyword(keywords);
        query.setPageNum(1);
        query.setPageSize(50);
        DatasetAPI.getList(query, RepositoryAdapters.wrap(new RepositoryCallback<PageResult<Dataset>>() {
            @Override
            public void onSuccess(PageResult<Dataset> data) {
                loading.setValue(false);
                List<Dataset> records = data != null ? data.getList() : null;
                datasetList.postValue(records != null ? records : new ArrayList<>());
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.setValue(false);
            }
        }));
    }
}
