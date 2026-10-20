package com.pei.clarity.sdk.api;

import com.pei.clarity.sdk.ApiCallback;
import com.pei.clarity.sdk.DehazeSDK;
import com.pei.clarity.sdk.model.Result;
import com.pei.clarity.sdk.model.billing.BalanceVO;

import retrofit2.Call;

/**
 * AI 计费API接口封装（用户端积分余额）
 */
public class AiBillingAPI {

    private AiBillingAPI() {
    }

    /**
     * 查询当前登录用户积分余额与已用量
     */
    public static void getBalance(ApiCallback<BalanceVO> callback) {
        Call<Result<BalanceVO>> call = DehazeSDK.getInstance().getAiBillingApiService().getBalance();
        call.enqueue(callback);
    }
}
