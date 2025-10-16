package com.pei.dehaze.sdk.api;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.DehazeSDK;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.model.PredParam;
import com.pei.dehaze.sdk.model.model.PredResult;
import com.pei.dehaze.sdk.model.model.EvalParam;
import com.pei.dehaze.sdk.model.model.EvalResult;

import java.util.List;

import retrofit2.Call;

/**
 * 模型相关API接口封装
 */
public class ModelAPI {

    /**
     * 模型预测
     *
     * @param data     预测参数
     * @param callback 回调函数
     */
    public static void prediction(PredParam data, ApiCallback<PredResult> callback) {
        Call<Result<PredResult>> call = DehazeSDK.getInstance().getModelApiService().modelPrediction(data);
        call.enqueue(callback);
    }

    /**
     * 模型评估
     *
     * @param data     评估参数
     * @param callback 回调函数
     */
    public static void evaluation(EvalParam data, ApiCallback<List<EvalResult>> callback) {
        Call<Result<List<EvalResult>>> call = DehazeSDK.getInstance().getModelApiService().modelEvaluation(data);
        call.enqueue(callback);
    }
}