package com.pei.dehaze.sdk.api;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.DehazeSDK;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.billing.BalanceVO;

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
