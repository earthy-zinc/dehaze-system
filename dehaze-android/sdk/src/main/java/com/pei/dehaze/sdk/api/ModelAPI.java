package com.pei.dehaze.sdk.api;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.DehazeSDK;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.evaluation.EvalParam;
import com.pei.dehaze.sdk.model.evaluation.EvalResult;
import com.pei.dehaze.sdk.model.evaluation.EvaluationLogVO;
import com.pei.dehaze.sdk.model.prediction.PredParam;
import com.pei.dehaze.sdk.model.prediction.PredictionLogVO;
import com.pei.dehaze.sdk.model.prediction.PredResult;

import retrofit2.Call;

/**
 * 预测/评估相关API接口封装
 */
public class ModelAPI {

    private ModelAPI() {
    }

    /**
     * 执行去雾预测
     *
     * @param data     预测参数
     * @param callback 回调函数
     */
    public static void predict(PredParam data, ApiCallback<PredResult> callback) {
        Call<Result<PredResult>> call = DehazeSDK.getInstance().getModelApiService().predict(data);
        call.enqueue(callback);
    }

    /**
     * 分页查询预测日志
     *
     * @param algorithmId 算法ID筛选（可传 null）
     * @param pageNum     页码
     * @param pageSize    每页数量
     * @param callback    回调函数
     */
    public static void listPredictionLogs(Long algorithmId, int pageNum, int pageSize,
                                          ApiCallback<PageResult<PredictionLogVO>> callback) {
        Call<Result<PageResult<PredictionLogVO>>> call = DehazeSDK.getInstance().getModelApiService()
                .listPredictionLogs(algorithmId, pageNum, pageSize);
        call.enqueue(callback);
    }

    /**
     * 执行效果评估
     *
     * @param data     评估参数
     * @param callback 回调函数
     */
    public static void evaluate(EvalParam data, ApiCallback<EvalResult> callback) {
        Call<Result<EvalResult>> call = DehazeSDK.getInstance().getModelApiService().evaluate(data);
        call.enqueue(callback);
    }

    /**
     * 分页查询评估日志
     *
     * @param algorithmId 算法ID筛选（可传 null）
     * @param pageNum     页码
     * @param pageSize    每页数量
     * @param callback    回调函数
     */
    public static void listEvaluationLogs(Long algorithmId, int pageNum, int pageSize,
                                          ApiCallback<PageResult<EvaluationLogVO>> callback) {
        Call<Result<PageResult<EvaluationLogVO>>> call = DehazeSDK.getInstance().getModelApiService()
                .listEvaluationLogs(algorithmId, pageNum, pageSize);
        call.enqueue(callback);
    }
}
