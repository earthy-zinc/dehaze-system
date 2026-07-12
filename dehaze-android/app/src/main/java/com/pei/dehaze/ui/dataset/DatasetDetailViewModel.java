package com.pei.dehaze.ui.dataset;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import com.pei.dehaze.repository.DatasetRepository;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.dataset.Dataset;
import com.pei.dehaze.sdk.model.dataset.DatasetImageFileInfo;
import com.pei.dehaze.sdk.model.dataset.DatasetItemCreateForm;
import com.pei.dehaze.sdk.model.dataset.DatasetItemUpdateForm;
import com.pei.dehaze.sdk.model.dataset.ImageItem;
import com.pei.dehaze.sdk.model.dataset.ImageItemQuery;
import com.pei.dehaze.sdk.model.dataset.ItemFileUpdateForm;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * 数据集详情页 ViewModel（数据集元数据 + 数据项分页 + 数据项 CRUD + 图片文件管理）
 */
public class DatasetDetailViewModel extends ViewModel {

    private final DatasetRepository repository;

    private final MutableLiveData<Dataset> datasetInfo = new MutableLiveData<>();
    private final MutableLiveData<List<ImageItem>> items = new MutableLiveData<>();
    private final MutableLiveData<Long> total = new MutableLiveData<>(0L);
    private final MutableLiveData<Boolean> loading = new MutableLiveData<>(false);
    private final MutableLiveData<String> error = new MutableLiveData<>();
    private final MutableLiveData<String> operationResult = new MutableLiveData<>();
    private final MutableLiveData<DatasetImageFileInfo> uploadedFile = new MutableLiveData<>();

    private long datasetId = 0;
    private int pageNum = 1;
    private final int pageSize = 10;
    private String keywords = "";
    private String sceneType;
    private String hazeLevel;

    /** 当前展示的图片类型（clear/hazy/depth/segment） */
    private String currentImageType = "hazy";

    public static final String[] IMAGE_TYPES = {"clear", "hazy", "depth", "segment"};

    public DatasetDetailViewModel() {
        repository = new DatasetRepository();
    }

    public void setDatasetId(long id) {
        this.datasetId = id;
    }

    public long getDatasetId() {
        return datasetId;
    }

    public String getCurrentImageType() {
        return currentImageType;
    }

    public void setCurrentImageType(String type) {
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

    public LiveData<Boolean> getLoading() {
        return loading;
    }

    public LiveData<String> getError() {
        return error;
    }

    public LiveData<String> getOperationResult() {
        return operationResult;
    }

    public LiveData<DatasetImageFileInfo> getUploadedFile() {
        return uploadedFile;
    }

    public void clearError() {
        error.setValue(null);
    }

    public void clearOperationResult() {
        operationResult.setValue(null);
    }

    public void clearUploadedFile() {
        uploadedFile.setValue(null);
    }

    /**
     * 加载数据集详情
     */
    public void loadDatasetInfo() {
        if (datasetId <= 0) return;
        loading.setValue(true);
        repository.getDatasetById(datasetId, new DatasetRepository.Callback<Dataset>() {
            @Override
            public void onSuccess(Dataset data) {
                datasetInfo.postValue(data);
                loading.postValue(false);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    /**
     * 加载数据项分页列表
     */
    public void loadItems() {
        if (datasetId <= 0) return;
        loading.setValue(true);
        ImageItemQuery query = buildQuery();
        repository.getItems(query, new DatasetRepository.Callback<PageResult<ImageItem>>() {
            @Override
            public void onSuccess(PageResult<ImageItem> data) {
                items.postValue(data != null && data.getList() != null ? data.getList() : new ArrayList<>());
                total.postValue(data != null ? data.getTotal() : 0L);
                loading.postValue(false);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
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
        loading.setValue(true);
        DatasetItemCreateForm form = new DatasetItemCreateForm();
        form.setDatasetId(datasetId);
        form.setName(name);
        repository.createItem(form, new DatasetRepository.Callback<Long>() {
            @Override
            public void onSuccess(Long data) {
                operationResult.postValue("新增数据项成功");
                loading.postValue(false);
                loadItems();
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    /**
     * 修改数据项名称
     */
    public void updateItem(long itemId, String name) {
        loading.setValue(true);
        DatasetItemUpdateForm form = new DatasetItemUpdateForm();
        form.setName(name);
        repository.updateItem(itemId, form, new DatasetRepository.Callback<Void>() {
            @Override
            public void onSuccess(Void data) {
                operationResult.postValue("修改数据项成功");
                loading.postValue(false);
                loadItems();
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    /**
     * 删除数据项
     */
    public void deleteItem(long itemId) {
        loading.setValue(true);
        repository.deleteItem(itemId, new DatasetRepository.Callback<Void>() {
            @Override
            public void onSuccess(Void data) {
                operationResult.postValue("删除数据项成功");
                loading.postValue(false);
                loadItems();
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    /**
     * 批量删除数据项
     */
    public void batchDeleteItems(List<Long> ids) {
        loading.setValue(true);
        repository.batchDeleteItems(ids, new DatasetRepository.Callback<Void>() {
            @Override
            public void onSuccess(Void data) {
                operationResult.postValue("批量删除成功");
                loading.postValue(false);
                loadItems();
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    /**
     * 上传数据项图片
     */
    public void uploadItemFile(long datasetItemId, String type, File file, String description) {
        loading.setValue(true);
        repository.uploadItemFile(datasetItemId, type, file, description,
                new DatasetRepository.Callback<DatasetImageFileInfo>() {
                    @Override
                    public void onSuccess(DatasetImageFileInfo data) {
                        operationResult.postValue("图片上传成功");
                        uploadedFile.postValue(data);
                        loading.postValue(false);
                        loadItems();
                    }

                    @Override
                    public void onError(String errorMessage) {
                        error.postValue(errorMessage);
                        loading.postValue(false);
                    }
                });
    }

    /**
     * 修改图片信息
     */
    public void updateItemFile(long fileId, ItemFileUpdateForm form) {
        loading.setValue(true);
        repository.updateItemFile(fileId, form, new DatasetRepository.Callback<Void>() {
            @Override
            public void onSuccess(Void data) {
                operationResult.postValue("修改图片信息成功");
                loading.postValue(false);
                loadItems();
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    /**
     * 删除图片
     */
    public void deleteItemFile(long fileId) {
        loading.setValue(true);
        repository.deleteItemFile(fileId, new DatasetRepository.Callback<Void>() {
            @Override
            public void onSuccess(Void data) {
                operationResult.postValue("删除图片成功");
                loading.postValue(false);
                loadItems();
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    /**
     * 批量删除图片
     */
    public void batchDeleteItemFiles(List<Long> ids) {
        loading.setValue(true);
        repository.batchDeleteItemFiles(ids, new DatasetRepository.Callback<Void>() {
            @Override
            public void onSuccess(Void data) {
                operationResult.postValue("批量删除图片成功");
                loading.postValue(false);
                loadItems();
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    /**
     * 刷新全部数据
     */
    public void refresh() {
        loadDatasetInfo();
        loadItems();
    }
}
