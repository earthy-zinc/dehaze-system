package com.pei.dehaze.ui.dataset;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.pei.dehaze.repository.DatasetRepository;
import com.pei.dehaze.repository.RepositoryCallback;
import com.pei.dehaze.ui.common.BaseViewModel;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.dataset.Dataset;
import com.pei.dehaze.sdk.model.dataset.DatasetQuery;

import java.util.ArrayList;
import java.util.List;

/**
 * 数据集列表页 ViewModel（树形列表 + CRUD）
 */
public class DatasetViewModel extends BaseViewModel {

    private final DatasetRepository repository;

    private final MutableLiveData<List<Dataset>> rootDatasets = new MutableLiveData<>();
    private final MutableLiveData<List<Dataset>> searchResults = new MutableLiveData<>();

    private boolean searchMode = false;
    private String keywords = "";

    private int searchPageNum = 1;
    private final int searchPageSize = 20;
    private long searchTotal = 0;

    public DatasetViewModel() {
        repository = new DatasetRepository();
    }

    public LiveData<List<Dataset>> getRootDatasets() {
        return rootDatasets;
    }

    public LiveData<List<Dataset>> getSearchResults() {
        return searchResults;
    }

    public boolean isSearchMode() {
        return searchMode;
    }

    /**
     * 加载根数据集
     */
    public void loadRoots() {
        searchMode = false;
        repository.getDatasetChildren(0, withLoading(data ->
                rootDatasets.postValue(data != null ? data : new ArrayList<>())));
    }

    /**
     * 懒加载子节点
     */
    public void loadChildren(long parentId, RepositoryCallback<List<Dataset>> callback) {
        repository.getDatasetChildren(parentId, callback);
    }

    /**
     * 搜索数据集（切换到搜索模式，显示扁平列表）
     */
    public void search(String keywords) {
        this.keywords = keywords == null ? "" : keywords.trim();
        searchMode = true;
        searchPageNum = 1;
        loadSearchPage();
    }

    public void searchNextPage() {
        long totalPages = (long) Math.ceil(searchTotal * 1.0 / searchPageSize);
        if (searchPageNum < totalPages) {
            searchPageNum++;
            loadSearchPage();
        }
    }

    private void loadSearchPage() {
        DatasetQuery query = new DatasetQuery();
        query.setPageNum(searchPageNum);
        query.setPageSize(searchPageSize);
        query.setKeywords(keywords);
        repository.getDatasetList(query, withLoading(data -> {
            searchTotal = data != null ? data.getTotal() : 0;
            List<Dataset> list = data != null ? data.getList() : new ArrayList<>();
            if (searchPageNum == 1) {
                searchResults.postValue(list);
            } else {
                List<Dataset> merged = new ArrayList<>(searchResults.getValue() != null
                        ? searchResults.getValue() : new ArrayList<>());
                merged.addAll(list);
                searchResults.postValue(merged);
            }
        }));
    }

    public long getSearchTotal() {
        return searchTotal;
    }

    /**
     * 清除搜索，回到树形模式
     */
    public void clearSearch() {
        keywords = "";
        searchMode = false;
        loadRoots();
    }

    /**
     * 新增数据集
     */
    public void addDataset(Dataset form) {
        repository.addDataset(form, withLoading(v -> {
            operationResult.postValue("新增数据集成功");
            reload();
        }));
    }

    /**
     * 修改数据集
     */
    public void updateDataset(long id, Dataset form) {
        repository.updateDataset(id, form, withLoading(v -> {
            operationResult.postValue("修改数据集成功");
            reload();
        }));
    }

    /**
     * 删除数据集
     */
    public void deleteDataset(long id) {
        repository.deleteDataset(id, withLoading(v -> {
            operationResult.postValue("删除数据集成功");
            reload();
        }));
    }

    /**
     * 批量删除数据集
     */
    public void batchDeleteDatasets(List<Long> ids) {
        repository.batchDeleteDatasets(ids, withLoading(v -> {
            operationResult.postValue("批量删除成功");
            reload();
        }));
    }

    private void reload() {
        if (searchMode) {
            search(keywords);
        } else {
            loadRoots();
        }
    }
}
