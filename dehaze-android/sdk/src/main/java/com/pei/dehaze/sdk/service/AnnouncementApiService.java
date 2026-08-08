package com.pei.dehaze.sdk.service;

import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.message.AnnouncementForm;
import com.pei.dehaze.sdk.model.message.AnnouncementSendResult;
import com.pei.dehaze.sdk.model.message.AnnouncementVO;

import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.DELETE;
import retrofit2.http.GET;
import retrofit2.http.PATCH;
import retrofit2.http.POST;
import retrofit2.http.PUT;
import retrofit2.http.Path;
import retrofit2.http.Query;

/**
 * 公告管理API服务接口
 * 对齐后端路由：/api/v1/announcements
 */
public interface AnnouncementApiService {

    @GET("/api/v1/announcements/page")
    Call<Result<PageResult<AnnouncementVO>>> getPage(
            @Query("pageNum") int pageNum,
            @Query("pageSize") int pageSize,
            @Query("title") String title,
            @Query("type") String type,
            @Query("status") Integer status);

    @POST("/api/v1/announcements")
    Call<Result<Object>> create(@Body AnnouncementForm form);

    @GET("/api/v1/announcements/{id}")
    Call<Result<AnnouncementVO>> getDetail(@Path("id") long id);

    @PUT("/api/v1/announcements/{id}")
    Call<Result<Void>> update(@Path("id") long id, @Body AnnouncementForm form);

    @DELETE("/api/v1/announcements/{id}")
    Call<Result<Void>> deleteById(@Path("id") long id);

    @POST("/api/v1/announcements/{id}/_send")
    Call<Result<AnnouncementSendResult>> send(@Path("id") long id);

    @PATCH("/api/v1/announcements/{id}/_cancel")
    Call<Result<Void>> cancel(@Path("id") long id);
}
