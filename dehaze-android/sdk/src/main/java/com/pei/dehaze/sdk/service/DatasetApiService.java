package com.pei.dehaze.sdk.service;

import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.dataset.BatchDeleteForm;
import com.pei.dehaze.sdk.model.dataset.Dataset;
import com.pei.dehaze.sdk.model.dataset.DatasetImageFileInfo;
import com.pei.dehaze.sdk.model.dataset.DatasetItemCreateForm;
import com.pei.dehaze.sdk.model.dataset.DatasetItemUpdateForm;
import com.pei.dehaze.sdk.model.dataset.ImageItem;
import com.pei.dehaze.sdk.model.dataset.ItemFileUpdateForm;

import okhttp3.MultipartBody;
import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.DELETE;
import retrofit2.http.GET;
import retrofit2.http.Multipart;
import retrofit2.http.POST;
import retrofit2.http.PUT;
import retrofit2.http.Part;
import retrofit2.http.Path;
import retrofit2.http.Query;

import java.util.List;

/**
 * 数据集相关API服务接口
 */
public interface DatasetApiService {

    // ===== 数据集 (/api/v1/datasets) =====

    @GET("/api/v1/datasets")
    Call<Result<PageResult<Dataset>>> getDatasetList(
            @Query("pageNum") int pageNum,
            @Query("pageSize") int pageSize,
            @Query("keywords") String keywords,
            @Query("type") String type,
            @Query("status") Integer status);

    @GET("/api/v1/datasets/tree")
    Call<Result<List<Dataset>>> getDatasetTree();

    @GET("/api/v1/datasets/options")
    Call<Result<List<Option>>> getDatasetOptions();

    @GET("/api/v1/datasets/children/{parentId}")
    Call<Result<List<Dataset>>> getDatasetChildren(@Path("parentId") long parentId);

    @GET("/api/v1/datasets/{id}")
    Call<Result<Dataset>> getDatasetById(@Path("id") long id);

    @POST("/api/v1/datasets")
    Call<Result<Void>> addDataset(@Body Dataset data);

    @PUT("/api/v1/datasets/{id}")
    Call<Result<Void>> updateDataset(@Path("id") long id, @Body Dataset data);

    @DELETE("/api/v1/datasets/{id}")
    Call<Result<Void>> deleteDataset(@Path("id") long id);

    @DELETE("/api/v1/datasets/batch")
    Call<Result<Void>> batchDeleteDatasets(@Body BatchDeleteForm form);

    // ===== 数据项 (/api/v1/dataset-items) =====

    @GET("/api/v1/dataset-items")
    Call<Result<PageResult<ImageItem>>> getDatasetItems(
            @Query("datasetId") long datasetId,
            @Query("pageNum") int pageNum,
            @Query("pageSize") int pageSize,
            @Query("keyword") String keyword,
            @Query("sceneType") String sceneType,
            @Query("hazeLevel") String hazeLevel);

    @GET("/api/v1/dataset-items/{itemId}")
    Call<Result<ImageItem>> getDatasetItemById(@Path("itemId") long itemId);

    @POST("/api/v1/dataset-items")
    Call<Result<Long>> createDatasetItem(@Body DatasetItemCreateForm form);

    @PUT("/api/v1/dataset-items/{itemId}")
    Call<Result<Void>> updateDatasetItem(@Path("itemId") long itemId, @Body DatasetItemUpdateForm form);

    @DELETE("/api/v1/dataset-items/{itemId}")
    Call<Result<Void>> deleteDatasetItem(@Path("itemId") long itemId);

    @DELETE("/api/v1/dataset-items/batch")
    Call<Result<Void>> batchDeleteDatasetItems(@Body BatchDeleteForm form);

    // ===== 图片文件 (/api/v1/item-files) =====

    @GET("/api/v1/item-files/{fileId}")
    Call<Result<DatasetImageFileInfo>> getItemFileById(@Path("fileId") long fileId);

    @Multipart
    @POST("/api/v1/item-files")
    Call<Result<DatasetImageFileInfo>> uploadItemFile(
            @Part MultipartBody.Part file,
            @Part("datasetItemId") okhttp3.RequestBody datasetItemId,
            @Part("type") okhttp3.RequestBody type,
            @Part("description") okhttp3.RequestBody description);

    @PUT("/api/v1/item-files/{fileId}")
    Call<Result<Void>> updateItemFile(@Path("fileId") long fileId, @Body ItemFileUpdateForm form);

    @DELETE("/api/v1/item-files/{fileId}")
    Call<Result<Void>> deleteItemFile(@Path("fileId") long fileId);

    @DELETE("/api/v1/item-files/batch")
    Call<Result<Void>> batchDeleteItemFiles(@Body BatchDeleteForm form);
}
