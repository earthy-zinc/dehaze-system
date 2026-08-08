package com.pei.dehaze.sdk.api;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.DehazeSDK;
import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.Result;
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

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.RequestBody;
import retrofit2.Call;

/**
 * 数据集相关API接口封装
 */
public class DatasetAPI {

    private DatasetAPI() {
    }

    // ===== 数据集 =====

    /**
     * 分页查询数据集列表
     */
    public static void getList(DatasetQuery query, ApiCallback<PageResult<Dataset>> callback) {
        Call<Result<PageResult<Dataset>>> call = DehazeSDK.getInstance().getDatasetApiService()
                .getDatasetList(query.getPageNum(), query.getPageSize(),
                        query.getKeyword(), query.getType(), query.getStatus());
        call.enqueue(callback);
    }

    /**
     * 获取完整数据集树
     */
    public static void getTree(ApiCallback<List<Dataset>> callback) {
        Call<Result<List<Dataset>>> call = DehazeSDK.getInstance().getDatasetApiService().getDatasetTree();
        call.enqueue(callback);
    }

    /**
     * 获取数据集下拉选项
     */
    public static void getOptions(ApiCallback<List<Option>> callback) {
        Call<Result<List<Option>>> call = DehazeSDK.getInstance().getDatasetApiService().getDatasetOptions();
        call.enqueue(callback);
    }

    /**
     * 获取子数据集列表（懒加载）
     */
    public static void getChildren(long parentId, ApiCallback<List<Dataset>> callback) {
        Call<Result<List<Dataset>>> call = DehazeSDK.getInstance().getDatasetApiService().getDatasetChildren(parentId);
        call.enqueue(callback);
    }

    /**
     * 根据ID获取数据集详情
     */
    public static void getDatasetInfoById(long id, ApiCallback<Dataset> callback) {
        Call<Result<Dataset>> call = DehazeSDK.getInstance().getDatasetApiService().getDatasetById(id);
        call.enqueue(callback);
    }

    /**
     * 新增数据集
     */
    public static void add(Dataset data, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getDatasetApiService().addDataset(data);
        call.enqueue(callback);
    }

    /**
     * 修改数据集
     */
    public static void update(long id, Dataset data, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getDatasetApiService().updateDataset(id, data);
        call.enqueue(callback);
    }

    /**
     * 删除单个数据集
     */
    public static void delete(long id, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getDatasetApiService().deleteDataset(id);
        call.enqueue(callback);
    }

    /**
     * 批量删除数据集
     */
    public static void batchDelete(BatchDeleteForm form, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getDatasetApiService().batchDeleteDatasets(form);
        call.enqueue(callback);
    }

    // ===== 数据项 =====

    /**
     * 分页查询数据项列表
     */
    public static void getItems(ImageItemQuery query, ApiCallback<PageResult<ImageItem>> callback) {
        Call<Result<PageResult<ImageItem>>> call = DehazeSDK.getInstance().getDatasetApiService()
                .getDatasetItems(query.getDatasetId(), query.getPageNum(), query.getPageSize(),
                        query.getKeyword(), query.getSceneType(), query.getHazeLevel());
        call.enqueue(callback);
    }

    /**
     * 获取数据项详情
     */
    public static void getItemById(long itemId, ApiCallback<ImageItem> callback) {
        Call<Result<ImageItem>> call = DehazeSDK.getInstance().getDatasetApiService().getDatasetItemById(itemId);
        call.enqueue(callback);
    }

    /**
     * 创建空数据项
     */
    public static void createItem(DatasetItemCreateForm form, ApiCallback<Long> callback) {
        Call<Result<Long>> call = DehazeSDK.getInstance().getDatasetApiService().createDatasetItem(form);
        call.enqueue(callback);
    }

    /**
     * 修改数据项
     */
    public static void updateItem(long itemId, DatasetItemUpdateForm form, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getDatasetApiService().updateDatasetItem(itemId, form);
        call.enqueue(callback);
    }

    /**
     * 删除数据项
     */
    public static void deleteItem(long itemId, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getDatasetApiService().deleteDatasetItem(itemId);
        call.enqueue(callback);
    }

    /**
     * 批量删除数据项
     */
    public static void batchDeleteItems(BatchDeleteForm form, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getDatasetApiService().batchDeleteDatasetItems(form);
        call.enqueue(callback);
    }

    // ===== 图片文件 =====

    /**
     * 获取图片文件详情
     */
    public static void getItemFileById(long fileId, ApiCallback<DatasetImageFileInfo> callback) {
        Call<Result<DatasetImageFileInfo>> call = DehazeSDK.getInstance().getDatasetApiService().getItemFileById(fileId);
        call.enqueue(callback);
    }

    /**
     * 上传数据项图片
     *
     * @param datasetItemId 数据项ID
     * @param type          图片类型(clear/hazy/trans)
     * @param file          图片文件
     * @param description   描述
     */
    public static void uploadItemFile(long datasetItemId, ImageType type, File file, String description,
                                       ApiCallback<DatasetImageFileInfo> callback) {
        RequestBody requestFile = RequestBody.create(MediaType.parse("image/*"), file);
        MultipartBody.Part filePart = MultipartBody.Part.createFormData("file", file.getName(), requestFile);
        RequestBody itemIdBody = RequestBody.create(MediaType.parse("text/plain"), String.valueOf(datasetItemId));
        RequestBody typeBody = RequestBody.create(MediaType.parse("text/plain"), type.getValue());
        RequestBody descBody = RequestBody.create(MediaType.parse("text/plain"), description != null ? description : "");

        Call<Result<DatasetImageFileInfo>> call = DehazeSDK.getInstance().getDatasetApiService()
                .uploadItemFile(filePart, itemIdBody, typeBody, descBody);
        call.enqueue(callback);
    }

    /**
     * 修改图片信息
     */
    public static void updateItemFile(long fileId, ItemFileUpdateForm form, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getDatasetApiService().updateItemFile(fileId, form);
        call.enqueue(callback);
    }

    /**
     * 删除图片
     */
    public static void deleteItemFile(long fileId, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getDatasetApiService().deleteItemFile(fileId);
        call.enqueue(callback);
    }

    /**
     * 批量删除图片
     */
    public static void batchDeleteItemFiles(BatchDeleteForm form, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getDatasetApiService().batchDeleteItemFiles(form);
        call.enqueue(callback);
    }
}
