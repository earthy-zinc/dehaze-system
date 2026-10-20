package com.pei.clarity.sdk.service;

import com.pei.clarity.sdk.model.Result;
import com.pei.clarity.sdk.model.apikey.ApiKeyCreateRequest;
import com.pei.clarity.sdk.model.apikey.ApiKeyInfo;

import java.util.List;

import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.DELETE;
import retrofit2.http.GET;
import retrofit2.http.POST;
import retrofit2.http.Path;

public interface ApiKeyApiService {

    @POST("/api/v1/auth/api-keys")
    Call<Result<ApiKeyInfo>> createApiKey(@Body ApiKeyCreateRequest request);

    @GET("/api/v1/auth/api-keys")
    Call<Result<List<ApiKeyInfo>>> listApiKeys();

    @DELETE("/api/v1/auth/api-keys/{id}")
    Call<Result<Void>> deleteApiKey(@Path("id") Long id);
}
