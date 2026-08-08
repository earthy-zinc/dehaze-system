package com.pei.dehaze.sdk.service;

import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.order.AutoRenewConfigForm;
import com.pei.dehaze.sdk.model.order.AutoRenewConfigVO;
import com.pei.dehaze.sdk.model.order.MyOrderVO;
import com.pei.dehaze.sdk.model.order.OrderCreateForm;
import com.pei.dehaze.sdk.model.order.OrderDetailVO;
import com.pei.dehaze.sdk.model.order.OrderPageVO;
import com.pei.dehaze.sdk.model.order.OrderStatsVO;
import com.pei.dehaze.sdk.model.order.PayRequest;
import com.pei.dehaze.sdk.model.order.PayResult;
import com.pei.dehaze.sdk.model.order.RefundApplyForm;
import com.pei.dehaze.sdk.model.order.RefundAuditForm;
import com.pei.dehaze.sdk.model.order.RefundRecordVO;

import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.GET;
import retrofit2.http.POST;
import retrofit2.http.PUT;
import retrofit2.http.Path;
import retrofit2.http.Query;

/**
 * 订单管理API服务接口
 * 对齐后端路由：/api/v1/orders
 */
public interface OrderApiService {

    @POST("/api/v1/orders")
    Call<Result<PayResult>> create(@Body OrderCreateForm form);

    @GET("/api/v1/orders/my")
    Call<Result<PageResult<MyOrderVO>>> listMy(
            @Query("pageNum") int pageNum,
            @Query("pageSize") int pageSize,
            @Query("status") String status);

    @GET("/api/v1/orders/{orderNo}")
    Call<Result<OrderDetailVO>> getDetail(@Path("orderNo") String orderNo);

    @PUT("/api/v1/orders/{orderNo}/cancel")
    Call<Result<Void>> cancel(@Path("orderNo") String orderNo, @Query("reason") String reason);

    @POST("/api/v1/orders/{orderNo}/pay")
    Call<Result<PayResult>> pay(@Path("orderNo") String orderNo, @Body PayRequest request);

    @POST("/api/v1/orders/{orderNo}/refund")
    Call<Result<Void>> applyRefund(@Path("orderNo") String orderNo, @Body RefundApplyForm form);

    @PUT("/api/v1/orders/auto-renew/config")
    Call<Result<Void>> updateAutoRenewConfig(@Body AutoRenewConfigForm form);

    @GET("/api/v1/orders/auto-renew/config")
    Call<Result<AutoRenewConfigVO>> getAutoRenewConfig(@Query("packageId") long packageId);

    @GET("/api/v1/orders/page")
    Call<Result<PageResult<OrderPageVO>>> getPage(
            @Query("pageNum") int pageNum,
            @Query("pageSize") int pageSize,
            @Query("orderNo") String orderNo,
            @Query("keywords") String keywords,
            @Query("status") String status,
            @Query("payMethod") String payMethod,
            @Query("amountMin") Double amountMin,
            @Query("amountMax") Double amountMax,
            @Query("paidTimeStart") String paidTimeStart,
            @Query("paidTimeEnd") String paidTimeEnd);

    @GET("/api/v1/orders/refunds/page")
    Call<Result<PageResult<RefundRecordVO>>> listRefunds(
            @Query("pageNum") int pageNum,
            @Query("pageSize") int pageSize,
            @Query("orderNo") String orderNo,
            @Query("keywords") String keywords,
            @Query("status") String status,
            @Query("applyTimeStart") String applyTimeStart,
            @Query("applyTimeEnd") String applyTimeEnd);

    @PUT("/api/v1/orders/refunds/{refundId}/approve")
    Call<Result<Void>> approveRefund(@Path("refundId") long refundId, @Body RefundAuditForm form);

    @PUT("/api/v1/orders/refunds/{refundId}/reject")
    Call<Result<Void>> rejectRefund(@Path("refundId") long refundId, @Body RefundAuditForm form);

    @GET("/api/v1/orders/stats")
    Call<Result<OrderStatsVO>> getStats(
            @Query("startTime") String startTime,
            @Query("endTime") String endTime);
}
