package com.pei.dehaze.ui.dataset;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.pei.dehaze.repository.DatasetRepository;
import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.repository.RepositoryCallback;
import com.pei.dehaze.sdk.api.DatasetAPI;
import com.pei.dehaze.ui.common.BaseLoadMoreViewModel;
import com.pei.dehaze.sdk.model.dataset.Dataset;
import com.pei.dehaze.sdk.model.dataset.DatasetQuery;

import java.util.ArrayList;
import java.util.List;

/**
 * 数据集列表页 ViewModel（树形列表 + CRUD）
 *
 * <p>两套数据：
 * <ul>
 *   <li>树形：{@link #rootDatasets} + {@link #loadChildren(long, RepositoryCallback)} 懒加载，不走分页基类</li>
 *   <li>搜索：复用 {@link BaseLoadMoreViewModel} 的 itemList/分页机制，{@link #loadPage()} 发起搜索请求</li>
 * </ul>
 */
public class DatasetViewModel extends BaseLoadMoreViewModel<Dataset> {

    private final DatasetRepository repository = new DatasetRepository();

    private final MutableLiveData<List<Dataset>> rootDatasets = new MutableLiveData<>();

    private boolean searchMode = false;
    private String keywords = "";

    public DatasetViewModel() {
        super(20);
    }

    public LiveData<List<Dataset>> getRootDatasets() {
        return rootDatasets;
    }

    public LiveData<List<Dataset>> getSearchResults() {
        return itemList;
    }

    public boolean isSearchMode() {
        return searchMode;
    }

    /**
     * 加载根数据集
     */
    public void loadRoots() {
        searchMode = false;
        DatasetAPI.getChildren(0, RepositoryAdapters.wrap(withLoading(data ->
                rootDatasets.postValue(data != null ? data : new ArrayList<>()))));
    }

    /**
     * 懒加载子节点
     */
    public void loadChildren(long parentId, RepositoryCallback<List<Dataset>> callback) {
        DatasetAPI.getChildren(parentId, RepositoryAdapters.wrap(callback));
    }

    /**
     * 搜索数据集（切换到搜索模式，显示扁平列表）
     */
    public void search(String keywords) {
        this.keywords = keywords == null ? "" : keywords.trim();
        searchMode = true;
        reload();
    }

    public void searchNextPage() {
        loadMore();
    }

    @Override
    protected void loadPage() {
        DatasetQuery query = new DatasetQuery();
        query.setPageNum(pageNum);
        query.setPageSize(pageSize);
        query.setKeyword(keywords);
        DatasetAPI.getList(query, RepositoryAdapters.wrap(withLoading(data ->
                onPageLoaded(data != null ? data.getList() : null,
                        data != null ? data.getTotal() : 0))));
    }

    public long getSearchTotal() {
        return total;
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
        DatasetAPI.add(form, RepositoryAdapters.wrap(withLoading(v -> {
            operationResult.postValue("新增数据集成功");
            refresh();
        })));
    }

    /**
     * 修改数据集
     */
    public void updateDataset(long id, Dataset form) {
        DatasetAPI.update(id, form, RepositoryAdapters.wrap(withLoading(v -> {
            operationResult.postValue("修改数据集成功");
            refresh();
        })));
    }

    /**
     * 删除数据集
     */
    public void deleteDataset(long id) {
        DatasetAPI.delete(id, RepositoryAdapters.wrap(withLoading(v -> {
            operationResult.postValue("删除数据集成功");
            refresh();
        })));
    }

    /**
     * 批量删除数据集
     */
    public void batchDeleteDatasets(List<Long> ids) {
        repository.batchDeleteDatasets(ids, withLoading(v -> {
            operationResult.postValue("批量删除成功");
            refresh();
        }));
    }

    /**
     * CRUD 后按当前模式刷新：搜索模式重新搜索，树形模式重载根节点。
     */
    private void refresh() {
        if (searchMode) {
            reload();
        } else {
            loadRoots();
        }
    }
}
