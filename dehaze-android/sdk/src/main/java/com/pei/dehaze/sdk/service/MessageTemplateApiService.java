package com.pei.dehaze.sdk.service;

import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.message.MessageTemplateForm;
import com.pei.dehaze.sdk.model.message.MessageTemplateVO;

import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.GET;
import retrofit2.http.PUT;
import retrofit2.http.Path;
import retrofit2.http.Query;

/**
 * 消息模板管理API服务接口
 * 对齐后端路由：/api/v1/message-templates
 */
public interface MessageTemplateApiService {

    @GET("/api/v1/message-templates/page")
    Call<Result<PageResult<MessageTemplateVO>>> getPage(
            @Query("pageNum") int pageNum,
            @Query("pageSize") int pageSize,
            @Query("name") String name,
            @Query("type") String type,
            @Query("status") Integer status);

    @GET("/api/v1/message-templates/{id}")
    Call<Result<MessageTemplateVO>> getDetail(@Path("id") long id);

    @PUT("/api/v1/message-templates/{id}")
    Call<Result<Void>> update(@Path("id") long id, @Body MessageTemplateForm form);
}
