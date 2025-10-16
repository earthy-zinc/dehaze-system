package com.pei.dehaze.sdk.api;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.DehazeSDK;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.dataset.Dataset;
import com.pei.dehaze.sdk.model.dataset.DatasetQuery;
import com.pei.dehaze.sdk.model.dataset.ImageItem;
import com.pei.dehaze.sdk.model.dataset.ImageItemQuery;
import com.pei.dehaze.sdk.model.dataset.DatasetImageFileInfo;

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

    /**
     * 数据集树形表格
     *
     * @param queryParams 查询参数
     * @param callback    回调函数
     */
    public static void getList(DatasetQuery queryParams, ApiCallback<List<Dataset>> callback) {
        Call<Result<List<Dataset>>> call = DehazeSDK.getInstance().getDatasetApiService().getDatasetList(queryParams.getKeywords());
        call.enqueue(callback);
    }

    /**
     * 获取数据集下拉列表
     *
     * @param callback 回调函数
     */
    public static void getOptions(ApiCallback<List<Option>> callback) {
        Call<Result<List<Option>>> call = DehazeSDK.getInstance().getDatasetApiService().getDatasetOptions();
        call.enqueue(callback);
    }

    /**
     * 根据Id获取数据集信息
     *
     * @param id       数据集id
     * @param callback 回调函数
     */
    public static void getDatasetInfoById(int id, ApiCallback<Dataset> callback) {
        Call<Result<Dataset>> call = DehazeSDK.getInstance().getDatasetApiService().getDatasetInfo(id);
        call.enqueue(callback);
    }

    /**
     * 获取数据集详细图片
     *
     * @param id          数据集ID
     * @param queryParams 查询参数
     * @param callback    回调函数
     */
    public static void getImageItem(int id, ImageItemQuery queryParams, ApiCallback<List<ImageItem>> callback) {
        Call<Result<List<ImageItem>>> call = DehazeSDK.getInstance().getDatasetApiService()
                .getDatasetImageItems(id, queryParams.getPageNum(), queryParams.getPageSize(), queryParams.getKeywords());
        call.enqueue(callback);
    }

    /**
     * 新增数据集
     *
     * @param data     数据集数据
     * @param callback 回调函数
     */
    public static void add(Dataset data, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getDatasetApiService().addDataset(data);
        call.enqueue(callback);
    }

    /**
     * 修改数据集
     *
     * @param id       数据集ID
     * @param data     数据集数据
     * @param callback 回调函数
     */
    public static void update(int id, Dataset data, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getDatasetApiService().updateDataset(id, data);
        call.enqueue(callback);
    }

    /**
     * 删除数据集
     *
     * @param ids      数据集ID列表
     * @param callback 回调函数
     */
    public static void deleteByIds(String ids, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getDatasetApiService().deleteDatasets(ids);
        call.enqueue(callback);
    }

    /**
     * 新增数据项
     *
     * @param datasetId 数据集ID
     * @param name      数据项名称
     * @param callback  回调函数
     */
    public static void addDatasetItem(int datasetId, String name, ApiCallback<Integer> callback) {
        Call<Result<Integer>> call = DehazeSDK.getInstance().getDatasetApiService().addDatasetItem(datasetId, name);
        call.enqueue(callback);
    }

    /**
     * 更新数据项
     *
     * @param datasetItemId 数据项ID
     * @param name          数据项名称
     * @param callback      回调函数
     */
    public static void updateDatasetItem(int datasetItemId, String name, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getDatasetApiService().updateDatasetItem(datasetItemId, name);
        call.enqueue(callback);
    }

    /**
     * 删除数据项
     *
     * @param datasetItemId 数据项ID
     * @param callback      回调函数
     */
    public static void deleteDatasetItem(int datasetItemId, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getDatasetApiService().deleteDatasetItem(datasetItemId);
        call.enqueue(callback);
    }

    /**
     * 上传数据项图片
     *
     * @param datasetId     数据集ID
     * @param datasetItemId 数据项ID
     * @param type          图片类型
     * @param file          文件
     * @param description   描述
     * @param callback      回调函数
     */
    public static void uploadItemImage(int datasetId, int datasetItemId, String type, 
                                       File file, String description, ApiCallback<DatasetImageFileInfo> callback) {
        // 创建MultipartBody.Part
        RequestBody requestFile = RequestBody.create(MediaType.parse("image/*"), file);
        MultipartBody.Part body = MultipartBody.Part.createFormData("file", file.getName(), requestFile);
        
        Call<Result<DatasetImageFileInfo>> call = DehazeSDK.getInstance().getDatasetApiService()
                .uploadDatasetItemImage(datasetId, datasetItemId, type, body, description);
        call.enqueue(callback);
    }

    /**
     * 更新数据项图片
     *
     * @param itemFileId  文件ID
     * @param type        图片类型
     * @param description 描述
     * @param callback    回调函数
     */
    public static void updateItemImage(int itemFileId, String type, String description, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getDatasetApiService().updateDatasetItemImage(itemFileId, type, description);
        call.enqueue(callback);
    }

    /**
     * 删除数据项图片
     *
     * @param itemFileId 文件ID
     * @param callback   回调函数
     */
    public static void deleteItemImage(int itemFileId, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getDatasetApiService().deleteDatasetItemImage(itemFileId);
        call.enqueue(callback);
    }
}