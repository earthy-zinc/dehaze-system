package com.pei.dehaze.sdk.api;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.DehazeSDK;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.favorite.FavoriteCountVO;
import com.pei.dehaze.sdk.model.favorite.FavoriteForm;
import com.pei.dehaze.sdk.model.favorite.FavoriteQuery;
import com.pei.dehaze.sdk.model.favorite.FavoriteStatusVO;
import com.pei.dehaze.sdk.model.favorite.FavoriteVO;

import java.util.List;
import java.util.stream.Collectors;

import retrofit2.Call;

/**
 * 收藏管理API接口封装
 * 对齐后端路由：/api/v1/favorites
 */
public class FavoriteAPI {

    private FavoriteAPI() {
    }

    /**
     * 收藏列表分页查询
     */
    public static void getPage(FavoriteQuery query, ApiCallback<PageResult<FavoriteVO>> callback) {
        Call<Result<PageResult<FavoriteVO>>> call = DehazeSDK.getInstance().getFavoriteApiService().getPage(
                query.getPageNum(),
                query.getPageSize(),
                query.getTargetType(),
                query.getKeywords(),
                query.getSortBy(),
                query.getSortOrder());
        call.enqueue(callback);
    }

    /**
     * 添加收藏
     */
    public static void add(FavoriteForm form, ApiCallback<Long> callback) {
        Call<Result<Long>> call = DehazeSDK.getInstance().getFavoriteApiService().add(form);
        call.enqueue(callback);
    }

    /**
     * 批量取消收藏
     */
    public static void deleteByIds(List<Long> ids, ApiCallback<Void> callback) {
        String joined = ids.stream().map(String::valueOf).collect(Collectors.joining(","));
        Call<Result<Void>> call = DehazeSDK.getInstance().getFavoriteApiService().deleteByIds(joined);
        call.enqueue(callback);
    }

    /**
     * 检查指定对象是否已收藏
     */
    public static void getStatus(long targetId, String targetType, ApiCallback<FavoriteStatusVO> callback) {
        Call<Result<FavoriteStatusVO>> call = DehazeSDK.getInstance().getFavoriteApiService().getStatus(targetId, targetType);
        call.enqueue(callback);
    }

    /**
     * 收藏数量统计
     */
    public static void getCount(String targetType, ApiCallback<List<FavoriteCountVO>> callback) {
        Call<Result<List<FavoriteCountVO>>> call = DehazeSDK.getInstance().getFavoriteApiService().getCount(targetType);
        call.enqueue(callback);
    }
}
