package com.pei.dehaze.sdk.service;

import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.favorite.FavoriteForm;
import com.pei.dehaze.sdk.model.favorite.FavoriteStatusVO;
import com.pei.dehaze.sdk.model.favorite.FavoriteVO;

import java.util.List;

import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.DELETE;
import retrofit2.http.GET;
import retrofit2.http.POST;
import retrofit2.http.Path;
import retrofit2.http.Query;

/**
 * 收藏管理API服务接口
 * 对齐后端路由：/api/v1/favorites
 */
public interface FavoriteApiService {
    /**
     * 收藏列表分页查询
     * GET /api/v1/favorites/page
     */
    @GET("/api/v1/favorites/page")
    Call<Result<PageResult<FavoriteVO>>> getPage(
            @Query("pageNum") int pageNum,
            @Query("pageSize") int pageSize,
            @Query("targetType") String targetType,
            @Query("keywords") String keywords,
            @Query("sortBy") String sortBy,
            @Query("sortOrder") String sortOrder);

    /**
     * 添加收藏
     * POST /api/v1/favorites
     */
    @POST("/api/v1/favorites")
    Call<Result<Long>> add(@Body FavoriteForm form);

    /**
     * 批量取消收藏
     * DELETE /api/v1/favorites/{ids}
     */
    @DELETE("/api/v1/favorites/{ids}")
    Call<Result<Void>> deleteByIds(@Path("ids") String ids);

    /**
     * 检查指定对象是否已收藏
     * GET /api/v1/favorites/{targetId}/status
     */
    @GET("/api/v1/favorites/{targetId}/status")
    Call<Result<FavoriteStatusVO>> getStatus(@Path("targetId") Long targetId, @Query("targetType") String targetType);

    /**
     * 收藏数量统计
     * GET /api/v1/favorites/count
     */
    @GET("/api/v1/favorites/count")
    Call<Result<List<com.pei.dehaze.sdk.model.favorite.FavoriteCountVO>>> getCount(@Query("targetType") String targetType);
}
