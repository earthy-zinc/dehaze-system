package com.pei.dehaze.sdk.service;

import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.algorithm.Algorithm;
import com.pei.dehaze.sdk.model.algorithm.AlgorithmQuery;

import java.util.List;

import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.GET;
import retrofit2.http.POST;
import retrofit2.http.PUT;
import retrofit2.http.DELETE;
import retrofit2.http.Query;
import retrofit2.http.Path;

/**
 * 算法相关API服务接口
 */
public interface AlgorithmApiService {
    // Algorithm APIs
    @GET("/api/v1/algorithms")
    Call<Result<List<Algorithm>>> getAlgorithmList(@Query("keywords") String keywords);

    @GET("/api/v1/algorithms/options")
    Call<Result<List<Option>>> getAlgorithmOptions();

    @GET("/api/v1/algorithms/{id}")
    Call<Result<Algorithm>> getAlgorithmInfo(@Path("id") int id);

    @POST("/api/v1/algorithms")
    Call<Result<Void>> addAlgorithm(@Body Algorithm data);

    @PUT("/api/v1/algorithms/{id}")
    Call<Result<Void>> updateAlgorithm(@Path("id") int id, @Body Algorithm data);

    @DELETE("/api/v1/algorithms")
    Call<Result<Void>> deleteAlgorithms(@Query("ids") String ids);
}
