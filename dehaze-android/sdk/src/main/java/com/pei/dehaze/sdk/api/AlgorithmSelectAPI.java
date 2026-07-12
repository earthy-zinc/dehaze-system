package com.pei.dehaze.sdk.api;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.DehazeSDK;
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

/**
 * 算法选择API接口封装（推荐/收藏/对比）
 */
public class AlgorithmSelectAPI {

    private AlgorithmSelectAPI() {
    }

    /**
     * 智能推荐算法
     *
     * @param imageUrl 待去雾图片URL
     * @param topN     推荐数量（1-10）
     * @param callback 回调
     */
    public static void recommend(String imageUrl, int topN, ApiCallback<List<AlgorithmRecommendVO>> callback) {
        RecommendRequest request = new RecommendRequest();
        request.setImageUrl(imageUrl);
        request.setTopN(topN);
        Call<Result<List<AlgorithmRecommendVO>>> call = DehazeSDK.getInstance().getAlgorithmSelectApiService().recommend(request);
        call.enqueue(callback);
    }

    /**
     * 收藏/取消收藏算法（切换状态）
     *
     * @param algorithmId 算法ID
     * @param callback    回调
     */
    public static void toggleFavorite(long algorithmId, ApiCallback<FavoriteToggleResult> callback) {
        FavoriteForm form = new FavoriteForm();
        form.setAlgorithmId(algorithmId);
        Call<Result<FavoriteToggleResult>> call = DehazeSDK.getInstance().getAlgorithmSelectApiService().toggleFavorite(form);
        call.enqueue(callback);
    }

    /**
     * 收藏列表
     *
     * @param callback 回调
     */
    public static void listFavorites(ApiCallback<List<FavoriteVO>> callback) {
        Call<Result<List<FavoriteVO>>> call = DehazeSDK.getInstance().getAlgorithmSelectApiService().listFavorites();
        call.enqueue(callback);
    }

    /**
     * 算法对比
     *
     * @param algorithmIds 算法ID列表（2-4个）
     * @param imageUrl     待对比的图片URL
     * @param callback     回调
     */
    public static void compare(List<Long> algorithmIds, String imageUrl, ApiCallback<List<AlgorithmCompareVO>> callback) {
        CompareRequest request = new CompareRequest();
        request.setAlgorithmIds(algorithmIds);
        request.setImageUrl(imageUrl);
        Call<Result<List<AlgorithmCompareVO>>> call = DehazeSDK.getInstance().getAlgorithmSelectApiService().compare(request);
        call.enqueue(callback);
    }
}
