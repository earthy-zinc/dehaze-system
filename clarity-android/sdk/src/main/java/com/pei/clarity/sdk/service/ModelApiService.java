package com.pei.clarity.sdk.service;

import com.pei.clarity.sdk.model.PageResult;
import com.pei.clarity.sdk.model.Result;
import com.pei.clarity.sdk.model.evaluation.EvalParam;
import com.pei.clarity.sdk.model.evaluation.EvalResult;
import com.pei.clarity.sdk.model.evaluation.EvaluationLogVO;

import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.GET;
import retrofit2.http.POST;
import retrofit2.http.Path;
import retrofit2.http.Query;

/**
 * 评估API服务接口
 * 对齐后端路由：/api/v1/evaluation
 */
public interface ModelApiService {

    @POST("/api/v1/evaluation")
    Call<Result<EvalResult>> evaluate(@Body EvalParam data);

    @GET("/api/v1/evaluation/{taskId}")
    Call<Result<EvalResult>> getEvalTaskStatus(@Path("taskId") long taskId);

    @GET("/api/v1/evaluation/logs")
    Call<Result<PageResult<EvaluationLogVO>>> listEvaluationLogs(
            @Query("modelId") String modelId,
            @Query("pageNum") int pageNum,
            @Query("pageSize") int pageSize);
}
