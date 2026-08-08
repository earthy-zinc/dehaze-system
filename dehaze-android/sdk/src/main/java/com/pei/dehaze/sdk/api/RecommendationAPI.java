package com.pei.dehaze.sdk.api;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.DehazeSDK;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.recommendation.AnalyzeForm;
import com.pei.dehaze.sdk.model.recommendation.ImageFeatureAnalysisVO;
import com.pei.dehaze.sdk.model.recommendation.RecommendationFeedback;
import com.pei.dehaze.sdk.model.recommendation.RecommendationReport;
import com.pei.dehaze.sdk.model.recommendation.RecommendationRule;
import com.pei.dehaze.sdk.model.recommendation.RecommendedAlgorithmVO;

import java.util.List;

import retrofit2.Call;

/**
 * 推荐管理API接口封装
 * 对齐后端路由：/api/v1/recommendations
 */
public class RecommendationAPI {

    private RecommendationAPI() {
    }

    public static void analyze(AnalyzeForm form, ApiCallback<ImageFeatureAnalysisVO> callback) {
        Call<Result<ImageFeatureAnalysisVO>> call = DehazeSDK.getInstance().getRecommendationApiService().analyze(form);
        call.enqueue(callback);
    }

    public static void getAlgorithmRecommendations(Long analysisId, String imageMd5, ApiCallback<List<RecommendedAlgorithmVO>> callback) {
        Call<Result<List<RecommendedAlgorithmVO>>> call = DehazeSDK.getInstance()
                .getRecommendationApiService().getAlgorithmRecommendations(analysisId, imageMd5);
        call.enqueue(callback);
    }

    public static void submitFeedback(RecommendationFeedback feedback, ApiCallback<Object> callback) {
        Call<Result<Object>> call = DehazeSDK.getInstance().getRecommendationApiService().submitFeedback(feedback);
        call.enqueue(callback);
    }

    public static void getRules(ApiCallback<List<RecommendationRule>> callback) {
        Call<Result<List<RecommendationRule>>> call = DehazeSDK.getInstance().getRecommendationApiService().getRules();
        call.enqueue(callback);
    }

    public static void updateRule(long id, RecommendationRule rule, ApiCallback<Object> callback) {
        Call<Result<Object>> call = DehazeSDK.getInstance().getRecommendationApiService().updateRule(id, rule);
        call.enqueue(callback);
    }

    public static void getReport(String startDate, String endDate, ApiCallback<RecommendationReport> callback) {
        Call<Result<RecommendationReport>> call = DehazeSDK.getInstance().getRecommendationApiService().getReport(startDate, endDate);
        call.enqueue(callback);
    }
}
