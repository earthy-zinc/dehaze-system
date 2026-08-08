package com.pei.dehaze.sdk.service;

import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.pkg.PackageDetailVO;
import com.pei.dehaze.sdk.model.pkg.PackageForm;
import com.pei.dehaze.sdk.model.pkg.PackagePageVO;
import com.pei.dehaze.sdk.model.pkg.PriceResult;
import com.pei.dehaze.sdk.model.pkg.SalesStatsVO;

import java.util.List;

import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.DELETE;
import retrofit2.http.GET;
import retrofit2.http.POST;
import retrofit2.http.PUT;
import retrofit2.http.Path;
import retrofit2.http.Query;

/**
 * 套餐管理API服务接口
 * 对齐后端路由：/api/v1/packages
 */
public interface PackageApiService {

    @GET("/api/v1/packages")
    Call<Result<List<PackageDetailVO>>> listOnSale();

    @GET("/api/v1/packages/{id}")
    Call<Result<PackageDetailVO>> getDetail(@Path("id") long id);

    @GET("/api/v1/packages/calculate-price")
    Call<Result<PriceResult>> calculatePrice(
            @Query("packageId") long packageId,
            @Query("userCouponId") Long userCouponId);

    @GET("/api/v1/packages/page")
    Call<Result<PageResult<PackagePageVO>>> getPage(
            @Query("pageNum") int pageNum,
            @Query("pageSize") int pageSize,
            @Query("name") String name,
            @Query("levelCode") String levelCode,
            @Query("period") String period,
            @Query("status") Integer status,
            @Query("startTime") String startTime,
            @Query("endTime") String endTime);

    @GET("/api/v1/packages/{id}/form")
    Call<Result<PackageForm>> getForm(@Path("id") long id);

    @POST("/api/v1/packages")
    Call<Result<Void>> add(@Body PackageForm form);

    @PUT("/api/v1/packages/{id}")
    Call<Result<Void>> update(@Path("id") long id, @Body PackageForm form);

    @PUT("/api/v1/packages/{id}/status")
    Call<Result<Void>> updateStatus(@Path("id") long id, @Query("status") int status);

    @DELETE("/api/v1/packages/{ids}")
    Call<Result<Void>> deleteByIds(@Path("ids") String ids);

    @GET("/api/v1/packages/sales/stats")
    Call<Result<SalesStatsVO>> getSalesStats();
}
