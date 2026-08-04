package com.pei.dehaze.sdk.api;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.DehazeSDK;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.recommendation.AnalyzeForm;
import com.pei.dehaze.sdk.model.recommendation.ImageFeatureAnalysisVO;
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

    /**
     * 图像特征分析
     */
    public static void analyze(AnalyzeForm form, ApiCallback<ImageFeatureAnalysisVO> callback) {
        Call<Result<ImageFeatureAnalysisVO>> call = DehazeSDK.getInstance().getRecommendationApiService().analyze(form);
        call.enqueue(callback);
    }

    /**
     * 获取算法推荐（分析后基于 imageMd5 查询）
     */
    public static void getAlgorithmRecommendations(Long analysisId, String imageMd5, ApiCallback<List<RecommendedAlgorithmVO>> callback) {
        Call<Result<List<RecommendedAlgorithmVO>>> call = DehazeSDK.getInstance()
                .getRecommendationApiService().getAlgorithmRecommendations(analysisId, imageMd5);
        call.enqueue(callback);
    }
}
