package com.pei.dehaze.sdk.service;

import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.algorithm_select.AlgorithmCompareVO;
import com.pei.dehaze.sdk.model.algorithm_select.AlgorithmRecommendVO;
import com.pei.dehaze.sdk.model.algorithm_select.CompareRequest;
import com.pei.dehaze.sdk.model.algorithm_select.FavoriteForm;
import com.pei.dehaze.sdk.model.algorithm_select.FavoriteToggleResult;
import com.pei.dehaze.sdk.model.algorithm_select.FavoriteVO;
import com.pei.dehaze.sdk.model.algorithm_select.RecommendRequest;

import java.util.List;

import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.GET;
import retrofit2.http.POST;

/**
 * 算法选择API服务接口
 * 对齐后端路由：/api/v1/algorithm-select（推荐/收藏/对比）
 */
public interface AlgorithmSelectApiService {
    /**
     * 智能推荐算法
     * POST /api/v1/algorithm-select/recommend
     */
    @POST("/api/v1/algorithm-select/recommend")
    Call<Result<List<AlgorithmRecommendVO>>> recommend(@Body RecommendRequest request);

    /**
     * 收藏/取消收藏算法（切换状态）
     * POST /api/v1/algorithm-select/favorite
     */
    @POST("/api/v1/algorithm-select/favorite")
    Call<Result<FavoriteToggleResult>> toggleFavorite(@Body FavoriteForm form);

    /**
     * 收藏列表
     * GET /api/v1/algorithm-select/favorites
     */
    @GET("/api/v1/algorithm-select/favorites")
    Call<Result<List<FavoriteVO>>> listFavorites();

    /**
     * 算法对比
     * POST /api/v1/algorithm-select/compare
     */
    @POST("/api/v1/algorithm-select/compare")
    Call<Result<List<AlgorithmCompareVO>>> compare(@Body CompareRequest request);
}
