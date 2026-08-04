package com.pei.dehaze.sdk.service;

import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.recommendation.AnalyzeForm;
import com.pei.dehaze.sdk.model.recommendation.ImageFeatureAnalysisVO;
import com.pei.dehaze.sdk.model.recommendation.RecommendedAlgorithmVO;

import java.util.List;

import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.GET;
import retrofit2.http.POST;
import retrofit2.http.Query;

/**
 * 推荐管理API服务接口
 * 对齐后端路由：/api/v1/recommendations
 */
public interface RecommendationApiService {
    /**
     * 图像特征分析
     * POST /api/v1/recommendations/analyze
     */
    @POST("/api/v1/recommendations/analyze")
    Call<Result<ImageFeatureAnalysisVO>> analyze(@Body AnalyzeForm form);

    /**
     * 获取算法推荐（分析后可基于 imageMd5 查询）
     * GET /api/v1/recommendations/algorithms
     */
    @GET("/api/v1/recommendations/algorithms")
    Call<Result<List<RecommendedAlgorithmVO>>> getAlgorithmRecommendations(
            @Query("analysisId") Long analysisId,
            @Query("imageMd5") String imageMd5);
}
