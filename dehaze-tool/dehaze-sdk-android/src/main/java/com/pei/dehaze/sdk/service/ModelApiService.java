package com.pei.dehaze.sdk.service;

import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.model.EvalParam;
import com.pei.dehaze.sdk.model.model.EvalResult;
import com.pei.dehaze.sdk.model.model.PredParam;
import com.pei.dehaze.sdk.model.model.PredResult;
import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.POST;

import java.util.List;

/**
 * 模型相关API服务接口
 */
public interface ModelApiService {
    // Model APIs
    @POST("/api/v1/model/prediction")
    Call<Result<PredResult>> modelPrediction(@Body PredParam data);

    @POST("/api/v1/model/evaluation")
    Call<Result<List<EvalResult>>> modelEvaluation(@Body EvalParam data);
}
