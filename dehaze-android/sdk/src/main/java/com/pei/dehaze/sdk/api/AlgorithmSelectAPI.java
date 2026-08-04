package com.pei.dehaze.sdk.api;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.DehazeSDK;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.algorithm_select.AlgorithmCompareForm;
import com.pei.dehaze.sdk.model.algorithm_select.AlgorithmCompareVO;
import com.pei.dehaze.sdk.model.algorithm_select.AlgorithmDetailVO;
import com.pei.dehaze.sdk.model.algorithm_select.AlgorithmSelectNodeVO;
import com.pei.dehaze.sdk.model.algorithm_select.AlgorithmTestForm;
import com.pei.dehaze.sdk.model.prediction.PredResult;

import java.util.List;

import retrofit2.Call;

/**
 * 算法选择API接口封装
 * 对齐后端路由：/api/v1/algorithms/select
 * 收藏请使用 {@link FavoriteAPI}（/api/v1/favorites），推荐请使用 {@link RecommendationAPI}（/api/v1/recommendations）
 */
public class AlgorithmSelectAPI {

    private AlgorithmSelectAPI() {
    }

    /**
     * 获取算法选择树（仅已发布算法）
     */
    public static void getTree(ApiCallback<List<AlgorithmSelectNodeVO>> callback) {
        Call<Result<List<AlgorithmSelectNodeVO>>> call = DehazeSDK.getInstance().getAlgorithmSelectApiService().getTree();
        call.enqueue(callback);
    }

    /**
     * 获取算法详情（含样例效果图、评分、使用次数）
     */
    public static void getDetail(long id, ApiCallback<AlgorithmDetailVO> callback) {
        Call<Result<AlgorithmDetailVO>> call = DehazeSDK.getInstance().getAlgorithmSelectApiService().getDetail(id);
        call.enqueue(callback);
    }

    /**
     * 上传自定义图片测试算法效果
     */
    public static void test(long id, AlgorithmTestForm form, ApiCallback<PredResult> callback) {
        Call<Result<PredResult>> call = DehazeSDK.getInstance().getAlgorithmSelectApiService().test(id, form);
        call.enqueue(callback);
    }

    /**
     * 搜索算法（关键词/拼音/标签）
     */
    public static void search(String keyword, ApiCallback<List<AlgorithmSelectNodeVO>> callback) {
        Call<Result<List<AlgorithmSelectNodeVO>>> call = DehazeSDK.getInstance().getAlgorithmSelectApiService().search(keyword);
        call.enqueue(callback);
    }

    /**
     * 算法对比（最多3个）
     */
    public static void compare(AlgorithmCompareForm form, ApiCallback<List<AlgorithmCompareVO>> callback) {
        Call<Result<List<AlgorithmCompareVO>>> call = DehazeSDK.getInstance().getAlgorithmSelectApiService().compare(form);
        call.enqueue(callback);
    }
}
