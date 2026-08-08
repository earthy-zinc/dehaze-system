package com.pei.dehaze.sdk.api;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.DehazeSDK;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.order.AutoRenewConfigForm;
import com.pei.dehaze.sdk.model.order.AutoRenewConfigVO;
import com.pei.dehaze.sdk.model.order.MyOrderQuery;
import com.pei.dehaze.sdk.model.order.MyOrderVO;
import com.pei.dehaze.sdk.model.order.OrderCreateForm;
import com.pei.dehaze.sdk.model.order.OrderDetailVO;
import com.pei.dehaze.sdk.model.order.OrderPageVO;
import com.pei.dehaze.sdk.model.order.OrderQuery;
import com.pei.dehaze.sdk.model.order.OrderStatsVO;
import com.pei.dehaze.sdk.model.order.PayRequest;
import com.pei.dehaze.sdk.model.order.PayResult;
import com.pei.dehaze.sdk.model.order.RefundApplyForm;
import com.pei.dehaze.sdk.model.order.RefundAuditForm;
import com.pei.dehaze.sdk.model.order.RefundQuery;
import com.pei.dehaze.sdk.model.order.RefundRecordVO;

import retrofit2.Call;

/**
 * 订单管理API接口封装
 * 对齐后端路由：/api/v1/orders
 */
public class OrderAPI {

    private OrderAPI() {
    }

    public static void create(OrderCreateForm form, ApiCallback<PayResult> callback) {
        Call<Result<PayResult>> call = DehazeSDK.getInstance().getOrderApiService().create(form);
        call.enqueue(callback);
    }

    public static void listMy(MyOrderQuery query, ApiCallback<PageResult<MyOrderVO>> callback) {
        Call<Result<PageResult<MyOrderVO>>> call = DehazeSDK.getInstance().getOrderApiService().listMy(
                query.getPageNum(), query.getPageSize(), query.getStatus());
        call.enqueue(callback);
    }

    public static void getDetail(String orderNo, ApiCallback<OrderDetailVO> callback) {
        Call<Result<OrderDetailVO>> call = DehazeSDK.getInstance().getOrderApiService().getDetail(orderNo);
        call.enqueue(callback);
    }

    public static void cancel(String orderNo, String reason, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getOrderApiService().cancel(orderNo, reason);
        call.enqueue(callback);
    }

    public static void pay(String orderNo, PayRequest request, ApiCallback<PayResult> callback) {
        Call<Result<PayResult>> call = DehazeSDK.getInstance().getOrderApiService().pay(orderNo, request);
        call.enqueue(callback);
    }

    public static void applyRefund(String orderNo, RefundApplyForm form, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getOrderApiService().applyRefund(orderNo, form);
        call.enqueue(callback);
    }

    public static void updateAutoRenewConfig(AutoRenewConfigForm form, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getOrderApiService().updateAutoRenewConfig(form);
        call.enqueue(callback);
    }

    public static void getAutoRenewConfig(long packageId, ApiCallback<AutoRenewConfigVO> callback) {
        Call<Result<AutoRenewConfigVO>> call = DehazeSDK.getInstance().getOrderApiService().getAutoRenewConfig(packageId);
        call.enqueue(callback);
    }

    public static void getPage(OrderQuery query, ApiCallback<PageResult<OrderPageVO>> callback) {
        Call<Result<PageResult<OrderPageVO>>> call = DehazeSDK.getInstance().getOrderApiService().getPage(
                query.getPageNum(), query.getPageSize(),
                query.getOrderNo(), query.getKeywords(), query.getStatus(), query.getPayMethod(),
                query.getAmountMin(), query.getAmountMax(),
                query.getPaidTimeStart(), query.getPaidTimeEnd());
        call.enqueue(callback);
    }

    public static void listRefunds(RefundQuery query, ApiCallback<PageResult<RefundRecordVO>> callback) {
        Call<Result<PageResult<RefundRecordVO>>> call = DehazeSDK.getInstance().getOrderApiService().listRefunds(
                query.getPageNum(), query.getPageSize(),
                query.getOrderNo(), query.getKeywords(), query.getStatus(),
                query.getApplyTimeStart(), query.getApplyTimeEnd());
        call.enqueue(callback);
    }

    public static void approveRefund(long refundId, RefundAuditForm form, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getOrderApiService().approveRefund(refundId, form);
        call.enqueue(callback);
    }

    public static void rejectRefund(long refundId, RefundAuditForm form, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getOrderApiService().rejectRefund(refundId, form);
        call.enqueue(callback);
    }

    public static void getStats(String startTime, String endTime, ApiCallback<OrderStatsVO> callback) {
        Call<Result<OrderStatsVO>> call = DehazeSDK.getInstance().getOrderApiService().getStats(startTime, endTime);
        call.enqueue(callback);
    }
}
