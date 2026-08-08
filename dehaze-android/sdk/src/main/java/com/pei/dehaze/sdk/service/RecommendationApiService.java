package com.pei.dehaze.sdk.service;

import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.recommendation.AnalyzeForm;
import com.pei.dehaze.sdk.model.recommendation.ImageFeatureAnalysisVO;
import com.pei.dehaze.sdk.model.recommendation.RecommendationFeedback;
import com.pei.dehaze.sdk.model.recommendation.RecommendationReport;
import com.pei.dehaze.sdk.model.recommendation.RecommendationRule;
import com.pei.dehaze.sdk.model.recommendation.RecommendedAlgorithmVO;

import java.util.List;

import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.GET;
import retrofit2.http.POST;
import retrofit2.http.PUT;
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

    /**
     * 提交推荐反馈
     * POST /api/v1/recommendations/feedback
     */
    @POST("/api/v1/recommendations/feedback")
    Call<Result<Object>> submitFeedback(@Body RecommendationFeedback feedback);

    /**
     * 获取推荐规则配置
     * GET /api/v1/recommendations/rules
     */
    @GET("/api/v1/recommendations/rules")
    Call<Result<List<RecommendationRule>>> getRules();

    /**
     * 更新推荐规则配置
     * PUT /api/v1/recommendations/rules
     */
    @PUT("/api/v1/recommendations/rules")
    Call<Result<Object>> updateRule(@Query("id") long id, @Body RecommendationRule rule);

    /**
     * 推荐效果报表
     * GET /api/v1/recommendations/report
     */
    @GET("/api/v1/recommendations/report")
    Call<Result<RecommendationReport>> getReport(
            @Query("startDate") String startDate,
            @Query("endDate") String endDate);
}
