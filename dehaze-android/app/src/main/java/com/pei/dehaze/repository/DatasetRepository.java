package com.pei.dehaze.repository;

import com.pei.dehaze.sdk.ApiCallback;
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
import com.pei.dehaze.sdk.model.dataset.ItemFileUpdateForm;
import com.pei.dehaze.sdk.network.ApiException;

import java.io.File;
import java.util.List;

public class DatasetRepository {

    public interface Callback<T> {
        void onSuccess(T data);
        void onError(String errorMessage);
    }

    // ===== 数据集 =====

    public void getDatasetList(DatasetQuery query, Callback<PageResult<Dataset>> callback) {
        DatasetAPI.getList(query, new ApiCallback<PageResult<Dataset>>() {
            @Override
            public void onSuccess(PageResult<Dataset> data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }

    public void getDatasetTree(Callback<List<Dataset>> callback) {
        DatasetAPI.getTree(new ApiCallback<List<Dataset>>() {
            @Override
            public void onSuccess(List<Dataset> data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }

    public void getDatasetOptions(Callback<List<Option>> callback) {
        DatasetAPI.getOptions(new ApiCallback<List<Option>>() {
            @Override
            public void onSuccess(List<Option> data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }

    public void getDatasetChildren(long parentId, Callback<List<Dataset>> callback) {
        DatasetAPI.getChildren(parentId, new ApiCallback<List<Dataset>>() {
            @Override
            public void onSuccess(List<Dataset> data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }

    public void getDatasetById(long id, Callback<Dataset> callback) {
        DatasetAPI.getDatasetInfoById(id, new ApiCallback<Dataset>() {
            @Override
            public void onSuccess(Dataset data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }

    public void addDataset(Dataset data, Callback<Void> callback) {
        DatasetAPI.add(data, new ApiCallback<Void>() {
            @Override
            public void onSuccess(Void data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }

    public void updateDataset(long id, Dataset data, Callback<Void> callback) {
        DatasetAPI.update(id, data, new ApiCallback<Void>() {
            @Override
            public void onSuccess(Void data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }

    public void deleteDataset(long id, Callback<Void> callback) {
        DatasetAPI.delete(id, new ApiCallback<Void>() {
            @Override
            public void onSuccess(Void data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }

    public void batchDeleteDatasets(List<Long> ids, Callback<Void> callback) {
        BatchDeleteForm form = new BatchDeleteForm();
        form.setIds(ids);
        DatasetAPI.batchDelete(form, new ApiCallback<Void>() {
            @Override
            public void onSuccess(Void data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }

    // ===== 数据项 =====

    public void getItems(ImageItemQuery query, Callback<PageResult<ImageItem>> callback) {
        DatasetAPI.getItems(query, new ApiCallback<PageResult<ImageItem>>() {
            @Override
            public void onSuccess(PageResult<ImageItem> data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }

    public void getItemById(long itemId, Callback<ImageItem> callback) {
        DatasetAPI.getItemById(itemId, new ApiCallback<ImageItem>() {
            @Override
            public void onSuccess(ImageItem data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }

    public void createItem(DatasetItemCreateForm form, Callback<Long> callback) {
        DatasetAPI.createItem(form, new ApiCallback<Long>() {
            @Override
            public void onSuccess(Long data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }

    public void updateItem(long itemId, DatasetItemUpdateForm form, Callback<Void> callback) {
        DatasetAPI.updateItem(itemId, form, new ApiCallback<Void>() {
            @Override
            public void onSuccess(Void data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }

    public void deleteItem(long itemId, Callback<Void> callback) {
        DatasetAPI.deleteItem(itemId, new ApiCallback<Void>() {
            @Override
            public void onSuccess(Void data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }

    public void batchDeleteItems(List<Long> ids, Callback<Void> callback) {
        BatchDeleteForm form = new BatchDeleteForm();
        form.setIds(ids);
        DatasetAPI.batchDeleteItems(form, new ApiCallback<Void>() {
            @Override
            public void onSuccess(Void data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }

    // ===== 图片文件 =====

    public void getItemFileById(long fileId, Callback<DatasetImageFileInfo> callback) {
        DatasetAPI.getItemFileById(fileId, new ApiCallback<DatasetImageFileInfo>() {
            @Override
            public void onSuccess(DatasetImageFileInfo data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }

    public void uploadItemFile(long datasetItemId, String type, File file, String description,
                               Callback<DatasetImageFileInfo> callback) {
        DatasetAPI.uploadItemFile(datasetItemId, type, file, description, new ApiCallback<DatasetImageFileInfo>() {
            @Override
            public void onSuccess(DatasetImageFileInfo data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }

    public void updateItemFile(long fileId, ItemFileUpdateForm form, Callback<Void> callback) {
        DatasetAPI.updateItemFile(fileId, form, new ApiCallback<Void>() {
            @Override
            public void onSuccess(Void data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }

    public void deleteItemFile(long fileId, Callback<Void> callback) {
        DatasetAPI.deleteItemFile(fileId, new ApiCallback<Void>() {
            @Override
            public void onSuccess(Void data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }

    public void batchDeleteItemFiles(List<Long> ids, Callback<Void> callback) {
        BatchDeleteForm form = new BatchDeleteForm();
        form.setIds(ids);
        DatasetAPI.batchDeleteItemFiles(form, new ApiCallback<Void>() {
            @Override
            public void onSuccess(Void data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }
}
