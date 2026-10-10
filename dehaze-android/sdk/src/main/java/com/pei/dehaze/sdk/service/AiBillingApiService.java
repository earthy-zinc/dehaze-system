package com.pei.dehaze.sdk.service;

import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.billing.BalanceVO;

import retrofit2.Call;
import retrofit2.http.GET;

/**
 * AI 计费API服务接口
 * 对齐后端路由：/api/v1/ai-billing
 */
public interface AiBillingApiService {

    /**
     * 查询当前登录用户积分余额与已用量
     * GET /api/v1/ai-billing/balance
     */
    @GET("/api/v1/ai-billing/balance")
    Call<Result<BalanceVO>> getBalance();
}
