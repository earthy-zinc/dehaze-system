package com.pei.dehaze.sdk.api;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.DehazeSDK;
import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.algorithm.Algorithm;
import com.pei.dehaze.sdk.model.algorithm.AlgorithmFavorite;
import com.pei.dehaze.sdk.model.algorithm.AlgorithmQuery;
import com.pei.dehaze.sdk.model.algorithm.AlgorithmStatus;
import com.pei.dehaze.sdk.model.algorithm.AlgorithmStatusForm;
import retrofit2.Call;

import java.util.List;
import java.util.stream.Collectors;

/**
 * 算法相关API接口封装
 */
public class AlgorithmAPI {

    private AlgorithmAPI() {
    }

    /**
     * 获取算法树形列表
     *
     * @param queryParams 查询参数
     * @param callback    回调函数
     */
    public static void getList(AlgorithmQuery queryParams, ApiCallback<List<Algorithm>> callback) {
        Call<Result<List<Algorithm>>> call = DehazeSDK.getInstance().getAlgorithmApiService().getAlgorithmList(queryParams.getKeywords());
        call.enqueue(callback);
    }

    /**
     * 算法对比
     *
     * @param ids      算法ID列表（英文逗号分割）
     * @param callback 回调函数
     */
    public static void compare(String ids, ApiCallback<List<Algorithm>> callback) {
        Call<Result<List<Algorithm>>> call = DehazeSDK.getInstance().getAlgorithmApiService().compareAlgorithms(ids);
        call.enqueue(callback);
    }

    /**
     * 获取模型下拉选项列表
     *
     * @param callback 回调函数
     */
    public static void getOption(ApiCallback<List<Option>> callback) {
        Call<Result<List<Option>>> call = DehazeSDK.getInstance().getAlgorithmApiService().getAlgorithmOptions();
        call.enqueue(callback);
    }

    /**
     * 获取用户收藏的算法列表
     *
     * @param callback 回调函数
     */
    public static void listFavorites(ApiCallback<List<AlgorithmFavorite>> callback) {
        Call<Result<List<AlgorithmFavorite>>> call = DehazeSDK.getInstance().getAlgorithmApiService().listFavorites();
        call.enqueue(callback);
    }

    /**
     * 获取算法详情
     *
     * @param id       算法ID
     * @param callback 回调函数
     */
    public static void getAlgorithmInfoById(long id, ApiCallback<Algorithm> callback) {
        Call<Result<Algorithm>> call = DehazeSDK.getInstance().getAlgorithmApiService().getAlgorithmInfo(id);
        call.enqueue(callback);
    }

    /**
     * 切换算法收藏状态
     *
     * @param id       算法ID
     * @param callback 回调函数
     */
    public static void toggleFavorite(long id, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getAlgorithmApiService().toggleFavorite(id);
        call.enqueue(callback);
    }

    /**
     * 新增算法
     *
     * @param data     算法数据
     * @param callback 回调函数
     */
    public static void add(Algorithm data, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getAlgorithmApiService().addAlgorithm(data);
        call.enqueue(callback);
    }

    /**
     * 修改算法
     *
     * @param id       算法ID
     * @param data     算法数据
     * @param callback 回调函数
     */
    public static void update(long id, Algorithm data, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getAlgorithmApiService().updateAlgorithm(id, data);
        call.enqueue(callback);
    }

    /**
     * 更新算法状态
     *
     * @param id       算法ID
     * @param status   状态
     * @param callback 回调函数
     */
    public static void updateStatus(long id, AlgorithmStatus status, ApiCallback<Void> callback) {
        AlgorithmStatusForm form = new AlgorithmStatusForm();
        form.setStatus(status);
        Call<Result<Void>> call = DehazeSDK.getInstance().getAlgorithmApiService().updateAlgorithmStatus(id, form);
        call.enqueue(callback);
    }

    /**
     * 删除算法
     *
     * @param ids      算法ID列表
     * @param callback 回调函数
     */
    public static void deleteByIds(List<Long> ids, ApiCallback<Void> callback) {
        String joined = ids.stream().map(String::valueOf).collect(Collectors.joining(","));
        Call<Result<Void>> call = DehazeSDK.getInstance().getAlgorithmApiService().deleteAlgorithms(joined);
        call.enqueue(callback);
    }
}
