package com.pei.dehaze.sdk.api;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.DehazeSDK;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.apikey.ApiKeyCreateRequest;
import com.pei.dehaze.sdk.model.apikey.ApiKeyInfo;

import java.util.List;

import retrofit2.Call;

public class ApiKeyAPI {

    private ApiKeyAPI() {
    }

    public static void create(ApiKeyCreateRequest request, ApiCallback<ApiKeyInfo> callback) {
        Call<Result<ApiKeyInfo>> call = DehazeSDK.getInstance().getApiKeyApiService().createApiKey(request);
        call.enqueue(callback);
    }

    public static void list(ApiCallback<List<ApiKeyInfo>> callback) {
        Call<Result<List<ApiKeyInfo>>> call = DehazeSDK.getInstance().getApiKeyApiService().listApiKeys();
        call.enqueue(callback);
    }

    public static void delete(Long id, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getApiKeyApiService().deleteApiKey(id);
        call.enqueue(callback);
    }
}
