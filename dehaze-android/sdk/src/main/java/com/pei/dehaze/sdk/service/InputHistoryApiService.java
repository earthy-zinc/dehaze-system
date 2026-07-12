package com.pei.dehaze.sdk.service;

import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.input_history.BatchDeleteForm;
import com.pei.dehaze.sdk.model.input_history.InputHistoryForm;
import com.pei.dehaze.sdk.model.input_history.InputHistoryUpdateForm;
import com.pei.dehaze.sdk.model.input_history.InputHistoryVO;
import com.pei.dehaze.sdk.model.input_history.SyncResultVO;

import java.util.List;

import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.DELETE;
import retrofit2.http.GET;
import retrofit2.http.POST;
import retrofit2.http.PUT;
import retrofit2.http.Path;
import retrofit2.http.Query;

/**
 * 图像输入历史API服务接口
 * 对齐后端路由：/api/v1/image-input/history
 */
public interface InputHistoryApiService {
    /**
     * 分页查询历史记录
     * GET /api/v1/image-input/history
     */
    @GET("/api/v1/image-input/history")
    Call<Result<PageResult<InputHistoryVO>>> listHistory(
            @Query("inputSource") String inputSource,
            @Query("favoriteOnly") Boolean favoriteOnly,
            @Query("keywords") String keywords,
            @Query("pageNum") int pageNum,
            @Query("pageSize") int pageSize);

    /**
     * 历史记录详情
     * GET /api/v1/image-input/history/{id}
     */
    @GET("/api/v1/image-input/history/{id}")
    Call<Result<InputHistoryVO>> getHistory(@Path("id") long id);

    /**
     * 创建历史记录
     * POST /api/v1/image-input/history
     */
    @POST("/api/v1/image-input/history")
    Call<Result<InputHistoryVO>> createHistory(@Body InputHistoryForm form);

    /**
     * 更新历史记录（如收藏、补充处理结果）
     * PUT /api/v1/image-input/history/{id}
     */
    @PUT("/api/v1/image-input/history/{id}")
    Call<Result<InputHistoryVO>> updateHistory(@Path("id") long id, @Body InputHistoryUpdateForm form);

    /**
     * 删除单条历史记录
     * DELETE /api/v1/image-input/history/{id}
     */
    @DELETE("/api/v1/image-input/history/{id}")
    Call<Result<Void>> deleteHistory(@Path("id") long id);

    /**
     * 批量删除历史记录
     * DELETE /api/v1/image-input/history/batch
     */
    @DELETE("/api/v1/image-input/history/batch")
    Call<Result<Void>> batchDeleteHistory(@Body BatchDeleteForm form);

    /**
     * 清空历史记录
     * DELETE /api/v1/image-input/history/clear
     */
    @DELETE("/api/v1/image-input/history/clear")
    Call<Result<Void>> clearHistory();

    /**
     * 同步本地与云端历史
     * POST /api/v1/image-input/history/sync
     */
    @POST("/api/v1/image-input/history/sync")
    Call<Result<SyncResultVO>> syncHistory(@Body List<InputHistoryForm> items);
}
