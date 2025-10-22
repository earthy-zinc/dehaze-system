package com.pei.dehaze.sdk.service;

import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.file.FileInfo;
import com.pei.dehaze.sdk.model.file.ImageFileInfo;

import okhttp3.MultipartBody;
import retrofit2.Call;
import retrofit2.http.GET;
import retrofit2.http.POST;
import retrofit2.http.DELETE;
import retrofit2.http.Query;
import retrofit2.http.Part;
import retrofit2.http.Multipart;

/**
 * 文件相关API服务接口
 */
public interface FileApiService {
    // File APIs
    @GET("/api/v1/files/check")
    Call<Result<Void>> checkFileUpload(@Query("md5") String md5);

    @Multipart
    @POST("/api/v1/files")
    Call<Result<FileInfo>> uploadFile(@Part MultipartBody.Part file, @Query("modelId") Integer modelId);

    @Multipart
    @POST("/api/v1/files")
    Call<Result<ImageFileInfo>> uploadImageFile(@Part MultipartBody.Part file, @Query("modelId") Integer modelId);

    @DELETE("/api/v1/files")
    Call<Result<Void>> deleteFileByPath(@Query("filePath") String filePath);
}
