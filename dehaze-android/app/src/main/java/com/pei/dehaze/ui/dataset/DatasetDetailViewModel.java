package com.pei.dehaze.ui.dataset;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.pei.dehaze.repository.DatasetRepository;
import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.sdk.api.DatasetAPI;
import com.pei.dehaze.ui.common.BaseViewModel;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.dataset.Dataset;
import com.pei.dehaze.sdk.model.dataset.DatasetImageFileInfo;
import com.pei.dehaze.sdk.model.dataset.DatasetItemCreateForm;
import com.pei.dehaze.sdk.model.dataset.DatasetItemUpdateForm;
import com.pei.dehaze.sdk.model.dataset.ImageItem;
import com.pei.dehaze.sdk.model.dataset.ImageItemQuery;
import com.pei.dehaze.sdk.model.dataset.ImageType;
import com.pei.dehaze.sdk.model.dataset.ItemFileUpdateForm;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * 数据集详情页 ViewModel（数据集元数据 + 数据项分页 + 数据项 CRUD + 图片文件管理）
 */
public class DatasetDetailViewModel extends BaseViewModel {

    private final DatasetRepository repository = new DatasetRepository();

    private final MutableLiveData<Dataset> datasetInfo = new MutableLiveData<>();
    private final MutableLiveData<List<ImageItem>> items = new MutableLiveData<>();
    private final MutableLiveData<Long> total = new MutableLiveData<>(0L);
    private final MutableLiveData<DatasetImageFileInfo> uploadedFile = new MutableLiveData<>();

    private long datasetId = 0;
    private int pageNum = 1;
    private final int pageSize = 10;
    private String keywords = "";
    private String sceneType;
    private String hazeLevel;

    /** 当前展示的图片类型（clear/hazy/trans） */
    private ImageType currentImageType = ImageType.HAZY;

    public void setDatasetId(long id) {
        this.datasetId = id;
    }

    public long getDatasetId() {
        return datasetId;
    }

    public ImageType getCurrentImageType() {
        return currentImageType;
    }

    public void setCurrentImageType(ImageType type) {
        this.currentImageType = type;
    }

    public LiveData<Dataset> getDatasetInfo() {
        return datasetInfo;
    }

    public LiveData<List<ImageItem>> getItems() {
        return items;
    }

    public LiveData<Long> getTotal() {
        return total;
    }

    public LiveData<DatasetImageFileInfo> getUploadedFile() {
        return uploadedFile;
    }

    public void clearUploadedFile() {
        uploadedFile.setValue(null);
    }

    /**
     * 加载数据集详情
     */
    public void loadDatasetInfo() {
        if (datasetId <= 0) return;
        DatasetAPI.getDatasetInfoById(datasetId, RepositoryAdapters.wrap(withLoading(datasetInfo::postValue)));
    }

    /**
     * 加载数据项分页列表
     */
    public void loadItems() {
        if (datasetId <= 0) return;
        ImageItemQuery query = buildQuery();
        DatasetAPI.getItems(query, RepositoryAdapters.wrap(withLoading(data -> {
            items.postValue(data != null ? data.getList() : new ArrayList<>());
            total.postValue(data != null ? data.getTotal() : 0L);
        })));
    }

    private ImageItemQuery buildQuery() {
        ImageItemQuery query = new ImageItemQuery();
        query.setDatasetId(datasetId);
        query.setPageNum(pageNum);
        query.setPageSize(pageSize);
        query.setKeywords(keywords);
        query.setSceneType(sceneType);
        query.setHazeLevel(hazeLevel);
        return query;
    }

    public void setQueryParams(String keywords, String sceneType, String hazeLevel) {
        this.keywords = keywords == null ? "" : keywords.trim();
        this.sceneType = sceneType;
        this.hazeLevel = hazeLevel;
        this.pageNum = 1;
    }

    public void resetQuery() {
        this.keywords = "";
        this.sceneType = null;
        this.hazeLevel = null;
        this.pageNum = 1;
    }

    public void nextPage() {
        long totalVal = total.getValue() != null ? total.getValue() : 0L;
        int totalPages = (int) Math.ceil(totalVal * 1.0 / pageSize);
        if (pageNum < totalPages) {
            pageNum++;
            loadItems();
        }
    }

    public void prevPage() {
        if (pageNum > 1) {
            pageNum--;
            loadItems();
        }
    }

    public int getPageNum() {
        return pageNum;
    }

    public int getPageSize() {
        return pageSize;
    }

    /**
     * 新建空数据项
     */
    public void createItem(String name) {
        DatasetItemCreateForm form = new DatasetItemCreateForm();
        form.setDatasetId(datasetId);
        form.setName(name);
        DatasetAPI.createItem(form, RepositoryAdapters.wrap(withLoading(v -> {
            operationResult.postValue("新增数据项成功");
            loadItems();
        })));
    }

    /**
     * 修改数据项名称
     */
    public void updateItem(long itemId, String name) {
        DatasetItemUpdateForm form = new DatasetItemUpdateForm();
        form.setName(name);
        DatasetAPI.updateItem(itemId, form, RepositoryAdapters.wrap(withLoading(v -> {
            operationResult.postValue("修改数据项成功");
            loadItems();
        })));
    }

    /**
     * 删除数据项
     */
    public void deleteItem(long itemId) {
        DatasetAPI.deleteItem(itemId, RepositoryAdapters.wrap(withLoading(v -> {
            operationResult.postValue("删除数据项成功");
            loadItems();
        })));
    }

    /**
     * 批量删除数据项
     */
    public void batchDeleteItems(List<Long> ids) {
        repository.batchDeleteItems(ids, withLoading(v -> {
            operationResult.postValue("批量删除成功");
            loadItems();
        }));
    }

    /**
     * 上传数据项图片
     */
    public void uploadItemFile(long datasetItemId, ImageType type, File file, String description) {
        DatasetAPI.uploadItemFile(datasetItemId, type, file, description,
                RepositoryAdapters.wrap(withLoading(data -> {
                    operationResult.postValue("图片上传成功");
                    uploadedFile.postValue(data);
                    loadItems();
                })));
    }

    /**
     * 修改图片信息
     */
    public void updateItemFile(long fileId, ItemFileUpdateForm form) {
        DatasetAPI.updateItemFile(fileId, form, RepositoryAdapters.wrap(withLoading(v -> {
            operationResult.postValue("修改图片信息成功");
            loadItems();
        })));
    }

    /**
     * 删除图片
     */
    public void deleteItemFile(long fileId) {
        DatasetAPI.deleteItemFile(fileId, RepositoryAdapters.wrap(withLoading(v -> {
            operationResult.postValue("删除图片成功");
            loadItems();
        })));
    }

    /**
     * 批量删除图片
     */
    public void batchDeleteItemFiles(List<Long> ids) {
        repository.batchDeleteItemFiles(ids, withLoading(v -> {
            operationResult.postValue("批量删除图片成功");
            loadItems();
        }));
    }

    /**
     * 刷新全部数据
     */
    public void refresh() {
        loadDatasetInfo();
        loadItems();
    }
}
