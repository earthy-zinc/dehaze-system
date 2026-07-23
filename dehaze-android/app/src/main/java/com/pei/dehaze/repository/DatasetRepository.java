package com.pei.dehaze.repository;

import com.pei.dehaze.sdk.api.DatasetAPI;
import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.dataset.BatchDeleteForm;
import com.pei.dehaze.sdk.model.dataset.Dataset;
import com.pei.dehaze.sdk.model.dataset.DatasetImageFileInfo;
import com.pei.dehaze.sdk.model.dataset.DatasetItemCreateForm;
import com.pei.dehaze.sdk.model.dataset.DatasetItemUpdateForm;
import com.pei.dehaze.sdk.model.dataset.DatasetQuery;
import com.pei.dehaze.sdk.model.dataset.ImageItem;
import com.pei.dehaze.sdk.model.dataset.ImageItemQuery;
import com.pei.dehaze.sdk.model.dataset.ImageType;
import com.pei.dehaze.sdk.model.dataset.ItemFileUpdateForm;

import java.io.File;
import java.util.List;

public class DatasetRepository {

    // ===== 数据集 =====

    public void getDatasetList(DatasetQuery query, RepositoryCallback<PageResult<Dataset>> callback) {
        DatasetAPI.getList(query, RepositoryAdapters.wrap(callback));
    }

    public void getDatasetTree(RepositoryCallback<List<Dataset>> callback) {
        DatasetAPI.getTree(RepositoryAdapters.wrap(callback));
    }

    public void getDatasetOptions(RepositoryCallback<List<Option>> callback) {
        DatasetAPI.getOptions(RepositoryAdapters.wrap(callback));
    }

    public void getDatasetChildren(long parentId, RepositoryCallback<List<Dataset>> callback) {
        DatasetAPI.getChildren(parentId, RepositoryAdapters.wrap(callback));
    }

    public void getDatasetById(long id, RepositoryCallback<Dataset> callback) {
        DatasetAPI.getDatasetInfoById(id, RepositoryAdapters.wrap(callback));
    }

    public void addDataset(Dataset data, RepositoryCallback<Void> callback) {
        DatasetAPI.add(data, RepositoryAdapters.wrap(callback));
    }

    public void updateDataset(long id, Dataset data, RepositoryCallback<Void> callback) {
        DatasetAPI.update(id, data, RepositoryAdapters.wrap(callback));
    }

    public void deleteDataset(long id, RepositoryCallback<Void> callback) {
        DatasetAPI.delete(id, RepositoryAdapters.wrap(callback));
    }

    public void batchDeleteDatasets(List<Long> ids, RepositoryCallback<Void> callback) {
        BatchDeleteForm form = new BatchDeleteForm();
        form.setIds(ids);
        DatasetAPI.batchDelete(form, RepositoryAdapters.wrap(callback));
    }

    // ===== 数据项 =====

    public void getItems(ImageItemQuery query, RepositoryCallback<PageResult<ImageItem>> callback) {
        DatasetAPI.getItems(query, RepositoryAdapters.wrap(callback));
    }

    public void getItemById(long itemId, RepositoryCallback<ImageItem> callback) {
        DatasetAPI.getItemById(itemId, RepositoryAdapters.wrap(callback));
    }

    public void createItem(DatasetItemCreateForm form, RepositoryCallback<Long> callback) {
        DatasetAPI.createItem(form, RepositoryAdapters.wrap(callback));
    }

    public void updateItem(long itemId, DatasetItemUpdateForm form, RepositoryCallback<Void> callback) {
        DatasetAPI.updateItem(itemId, form, RepositoryAdapters.wrap(callback));
    }

    public void deleteItem(long itemId, RepositoryCallback<Void> callback) {
        DatasetAPI.deleteItem(itemId, RepositoryAdapters.wrap(callback));
    }

    public void batchDeleteItems(List<Long> ids, RepositoryCallback<Void> callback) {
        BatchDeleteForm form = new BatchDeleteForm();
        form.setIds(ids);
        DatasetAPI.batchDeleteItems(form, RepositoryAdapters.wrap(callback));
    }

    // ===== 图片文件 =====

    public void getItemFileById(long fileId, RepositoryCallback<DatasetImageFileInfo> callback) {
        DatasetAPI.getItemFileById(fileId, RepositoryAdapters.wrap(callback));
    }

    public void uploadItemFile(long datasetItemId, ImageType type, File file, String description,
                               RepositoryCallback<DatasetImageFileInfo> callback) {
        DatasetAPI.uploadItemFile(datasetItemId, type, file, description, RepositoryAdapters.wrap(callback));
    }

    public void updateItemFile(long fileId, ItemFileUpdateForm form, RepositoryCallback<Void> callback) {
        DatasetAPI.updateItemFile(fileId, form, RepositoryAdapters.wrap(callback));
    }

    public void deleteItemFile(long fileId, RepositoryCallback<Void> callback) {
        DatasetAPI.deleteItemFile(fileId, RepositoryAdapters.wrap(callback));
    }

    public void batchDeleteItemFiles(List<Long> ids, RepositoryCallback<Void> callback) {
        BatchDeleteForm form = new BatchDeleteForm();
        form.setIds(ids);
        DatasetAPI.batchDeleteItemFiles(form, RepositoryAdapters.wrap(callback));
    }
}
