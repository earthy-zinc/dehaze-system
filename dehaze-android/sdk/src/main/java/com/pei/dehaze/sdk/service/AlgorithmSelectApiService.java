package com.pei.dehaze.sdk.service;

import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.algorithm_select.AlgorithmCompareForm;
import com.pei.dehaze.sdk.model.algorithm_select.AlgorithmCompareVO;
import com.pei.dehaze.sdk.model.algorithm_select.AlgorithmDetailVO;
import com.pei.dehaze.sdk.model.algorithm_select.AlgorithmSelectNodeVO;
import com.pei.dehaze.sdk.model.algorithm_select.AlgorithmTestForm;
import com.pei.dehaze.sdk.model.prediction.PredResult;

import java.util.List;

import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.GET;
import retrofit2.http.POST;
import retrofit2.http.Path;
import retrofit2.http.Query;

/**
 * 算法选择API服务接口
 * 对齐后端路由：/api/v1/algorithms/select
 */
public interface AlgorithmSelectApiService {
    /**
     * 获取算法选择树（仅已发布算法）
     * GET /api/v1/algorithms/select/tree
     */
    @GET("/api/v1/algorithms/select/tree")
    Call<Result<List<AlgorithmSelectNodeVO>>> getTree();

    /**
     * 获取算法详情（含样例效果图、评分、使用次数）
     * GET /api/v1/algorithms/select/{id}
     */
    @GET("/api/v1/algorithms/select/{id}")
    Call<Result<AlgorithmDetailVO>> getDetail(@Path("id") Long id);

    /**
     * 上传自定义图片测试算法效果
     * POST /api/v1/algorithms/select/{id}/test
     */
    @POST("/api/v1/algorithms/select/{id}/test")
    Call<Result<PredResult>> test(@Path("id") Long id, @Body AlgorithmTestForm form);

    /**
     * 搜索算法（关键词/拼音/标签）
     * GET /api/v1/algorithms/select/search
     */
    @GET("/api/v1/algorithms/select/search")
    Call<Result<List<AlgorithmSelectNodeVO>>> search(@Query("keyword") String keyword);

    /**
     * 算法对比（最多3个）
     * POST /api/v1/algorithms/select/compare
     */
    @POST("/api/v1/algorithms/select/compare")
    Call<Result<List<AlgorithmCompareVO>>> compare(@Body AlgorithmCompareForm form);
}
