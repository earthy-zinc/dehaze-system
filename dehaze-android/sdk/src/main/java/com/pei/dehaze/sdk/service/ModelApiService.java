package com.pei.dehaze.sdk.service;

import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.evaluation.EvalParam;
import com.pei.dehaze.sdk.model.evaluation.EvalResult;
import com.pei.dehaze.sdk.model.evaluation.EvaluationLogVO;
import com.pei.dehaze.sdk.model.prediction.PredParam;
import com.pei.dehaze.sdk.model.prediction.PredictionLogVO;
import com.pei.dehaze.sdk.model.prediction.PredResult;

import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.GET;
import retrofit2.http.POST;
import retrofit2.http.Path;
import retrofit2.http.Query;

/**
 * 预测/评估相关API服务接口
 */
public interface ModelApiService {

    @POST("/api/v1/prediction")
    Call<Result<PredResult>> predict(@Body PredParam data);

    @GET("/api/v1/prediction/{taskId}")
    Call<Result<PredResult>> getPredTaskStatus(@Path("taskId") long taskId);

    @GET("/api/v1/prediction/logs")
    Call<Result<PageResult<PredictionLogVO>>> listPredictionLogs(
            @Query("algorithmId") Long algorithmId,
            @Query("pageNum") int pageNum,
            @Query("pageSize") int pageSize);

    @POST("/api/v1/evaluation")
    Call<Result<EvalResult>> evaluate(@Body EvalParam data);

    @GET("/api/v1/evaluation/{taskId}")
    Call<Result<EvalResult>> getEvalTaskStatus(@Path("taskId") long taskId);

    @GET("/api/v1/evaluation/logs")
    Call<Result<PageResult<EvaluationLogVO>>> listEvaluationLogs(
            @Query("algorithmId") Long algorithmId,
            @Query("pageNum") int pageNum,
            @Query("pageSize") int pageSize);
}
