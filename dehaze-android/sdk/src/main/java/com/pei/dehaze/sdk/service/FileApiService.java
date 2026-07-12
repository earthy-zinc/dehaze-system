package com.pei.dehaze.sdk.service;

import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.file.FileInfo;

import okhttp3.MultipartBody;
import okhttp3.ResponseBody;
import retrofit2.Call;
import retrofit2.http.DELETE;
import retrofit2.http.GET;
import retrofit2.http.Multipart;
import retrofit2.http.POST;
import retrofit2.http.Part;
import retrofit2.http.Path;
import retrofit2.http.Query;

/**
 * 文件相关API服务接口
 */
public interface FileApiService {

    @GET("/api/v1/files/check")
    Call<Result<Boolean>> checkFile(@Query("md5") String md5);

    @GET("/api/v1/files/page")
    Call<Result<PageResult<FileInfo>>> getFilePage(
            @Query("pageNum") int pageNum,
            @Query("pageSize") int pageSize,
            @Query("keywords") String keywords);

    @GET("/api/v1/files/download/{objectName}")
    Call<ResponseBody> downloadFile(@Path("objectName") String objectName);

    @GET("/api/v1/files/{fileId}")
    Call<Result<FileInfo>> getFileDetail(@Path("fileId") long fileId);

    @Multipart
    @POST("/api/v1/files")
    Call<Result<FileInfo>> uploadFile(@Part MultipartBody.Part file);

    @DELETE("/api/v1/files")
    Call<Result<Void>> deleteFile(@Query("fileId") long fileId);
}
