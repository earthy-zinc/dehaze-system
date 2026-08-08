package com.pei.dehaze.sdk.service;

import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.message.MessageSendRequest;
import com.pei.dehaze.sdk.model.message.MessageSendResult;
import com.pei.dehaze.sdk.model.message.MessageVO;
import com.pei.dehaze.sdk.model.message.ReadAllResult;
import com.pei.dehaze.sdk.model.message.UnreadCountVO;

import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.DELETE;
import retrofit2.http.GET;
import retrofit2.http.PATCH;
import retrofit2.http.POST;
import retrofit2.http.Path;
import retrofit2.http.Query;

/**
 * 消息管理API服务接口
 * 对齐后端路由：/api/v1/messages
 */
public interface MessageApiService {

    @GET("/api/v1/messages")
    Call<Result<PageResult<MessageVO>>> getPage(
            @Query("pageNum") int pageNum,
            @Query("pageSize") int pageSize,
            @Query("type") String type,
            @Query("readStatus") Integer readStatus);

    @GET("/api/v1/messages/unread-count")
    Call<Result<UnreadCountVO>> getUnreadCount();

    @GET("/api/v1/messages/{id}")
    Call<Result<MessageVO>> getDetail(@Path("id") long id);

    @PATCH("/api/v1/messages/{id}/_read")
    Call<Result<Void>> markRead(@Path("id") long id);

    @PATCH("/api/v1/messages/_read-all")
    Call<Result<ReadAllResult>> markAllRead(@Query("type") String type);

    @DELETE("/api/v1/messages/{ids}")
    Call<Result<Void>> deleteByIds(@Path("ids") String ids);

    @GET("/api/v1/messages/search")
    Call<Result<PageResult<MessageVO>>> search(
            @Query("pageNum") int pageNum,
            @Query("pageSize") int pageSize,
            @Query("keyword") String keyword);

    @POST("/api/v1/messages/send")
    Call<Result<MessageSendResult>> send(@Body MessageSendRequest request);
}
