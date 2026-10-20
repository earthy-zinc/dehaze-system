package com.pei.clarity.sdk.service;

import com.pei.clarity.sdk.model.Result;
import com.pei.clarity.sdk.model.ai_model.AiModelVO;

import java.util.List;

import retrofit2.Call;
import retrofit2.http.GET;
import retrofit2.http.Query;

/**
 * AI 模型目录API服务接口
 * 对齐后端路由：/api/v1/ai/models
 */
public interface AiModelApiService {

    /**
     * 启用模型列表（按登录用户 VIP 等级过滤，无特殊权限）
     * GET /api/v1/ai/models/enabled
     */
    @GET("/api/v1/ai/models/enabled")
    Call<Result<List<AiModelVO>>> listEnabled(
            @Query("modelType") String modelType,
            @Query("taskTags") String taskTags);
}
