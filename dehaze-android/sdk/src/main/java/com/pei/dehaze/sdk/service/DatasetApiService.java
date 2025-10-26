package com.pei.dehaze.sdk.service;

import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.dataset.Dataset;
import com.pei.dehaze.sdk.model.dataset.DatasetImageFileInfo;
import com.pei.dehaze.sdk.model.dataset.ImageItem;
import okhttp3.MultipartBody;
import retrofit2.Call;
import retrofit2.http.*;

import java.util.List;

/**
 * 数据集相关API服务接口
 */
public interface DatasetApiService {
    // Dataset APIs
    @GET("/api/v1/dataset")
    Call<Result<List<Dataset>>> getDatasetList(@Query("keywords") String keywords);

    @GET("/api/v1/dataset/options")
    Call<Result<List<Option>>> getDatasetOptions();

    @GET("/api/v1/dataset/{id}")
    Call<Result<Dataset>> getDatasetInfo(@Path("id") int id);

    @GET("/api/v1/dataset/{id}/images")
    Call<Result<List<ImageItem>>> getDatasetImageItems(@Path("id") int id,
                                                       @Query("pageNum") int pageNum,
                                                       @Query("pageSize") int pageSize,
                                                       @Query("keywords") String keywords);

    @POST("/api/v1/dataset")
    Call<Result<Void>> addDataset(@Body Dataset data);

    @PUT("/api/v1/dataset/{id}")
    Call<Result<Void>> updateDataset(@Path("id") int id, @Body Dataset data);

    @DELETE("/api/v1/dataset/{ids}")
    Call<Result<Void>> deleteDatasets(@Path("ids") String ids);

    @POST("/api/v1/dataset/item")
    Call<Result<Integer>> addDatasetItem(@Query("datasetId") int datasetId, @Query("name") String name);

    @PUT("/api/v1/dataset/item")
    Call<Result<Void>> updateDatasetItem(@Query("datasetItemId") int datasetItemId, @Query("name") String name);

    @DELETE("/api/v1/dataset/item")
    Call<Result<Void>> deleteDatasetItem(@Query("datasetItemId") int datasetItemId);

    @Multipart
    @POST("/api/v1/dataset/image")
    Call<Result<DatasetImageFileInfo>> uploadDatasetItemImage(@Query("datasetId") int datasetId,
                                                              @Query("datasetItemId") int datasetItemId,
                                                              @Query("type") String type,
                                                              @Part MultipartBody.Part file,
                                                              @Query("description") String description);

    @PUT("/api/v1/dataset/image")
    Call<Result<Void>> updateDatasetItemImage(@Query("itemFileId") int itemFileId,
                                              @Query("type") String type,
                                              @Query("description") String description);

    @DELETE("/api/v1/dataset/image")
    Call<Result<Void>> deleteDatasetItemImage(@Query("itemFileId") int itemFileId);
}
