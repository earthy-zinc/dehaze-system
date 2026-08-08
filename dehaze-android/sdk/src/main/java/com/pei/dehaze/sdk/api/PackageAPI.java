package com.pei.dehaze.sdk.api;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.DehazeSDK;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.pkg.PackageDetailVO;
import com.pei.dehaze.sdk.model.pkg.PackageForm;
import com.pei.dehaze.sdk.model.pkg.PackagePageVO;
import com.pei.dehaze.sdk.model.pkg.PackageQuery;
import com.pei.dehaze.sdk.model.pkg.PriceResult;
import com.pei.dehaze.sdk.model.pkg.SalesStatsVO;

import java.util.List;

import retrofit2.Call;

/**
 * 套餐管理API接口封装
 * 对齐后端路由：/api/v1/packages
 */
public class PackageAPI {

    private PackageAPI() {
    }

    public static void listOnSale(ApiCallback<List<PackageDetailVO>> callback) {
        Call<Result<List<PackageDetailVO>>> call = DehazeSDK.getInstance().getPackageApiService().listOnSale();
        call.enqueue(callback);
    }

    public static void getDetail(long id, ApiCallback<PackageDetailVO> callback) {
        Call<Result<PackageDetailVO>> call = DehazeSDK.getInstance().getPackageApiService().getDetail(id);
        call.enqueue(callback);
    }

    public static void calculatePrice(long packageId, Long userCouponId, ApiCallback<PriceResult> callback) {
        Call<Result<PriceResult>> call = DehazeSDK.getInstance().getPackageApiService().calculatePrice(packageId, userCouponId);
        call.enqueue(callback);
    }

    public static void getPage(PackageQuery query, ApiCallback<PageResult<PackagePageVO>> callback) {
        Call<Result<PageResult<PackagePageVO>>> call = DehazeSDK.getInstance().getPackageApiService().getPage(
                query.getPageNum(), query.getPageSize(),
                query.getName(), query.getLevelCode(), query.getPeriod(),
                query.getStatus(), query.getStartTime(), query.getEndTime());
        call.enqueue(callback);
    }

    public static void getForm(long id, ApiCallback<PackageForm> callback) {
        Call<Result<PackageForm>> call = DehazeSDK.getInstance().getPackageApiService().getForm(id);
        call.enqueue(callback);
    }

    public static void add(PackageForm form, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getPackageApiService().add(form);
        call.enqueue(callback);
    }

    public static void update(long id, PackageForm form, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getPackageApiService().update(id, form);
        call.enqueue(callback);
    }

    public static void updateStatus(long id, int status, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getPackageApiService().updateStatus(id, status);
        call.enqueue(callback);
    }

    public static void deleteByIds(String ids, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getPackageApiService().deleteByIds(ids);
        call.enqueue(callback);
    }

    public static void getSalesStats(ApiCallback<SalesStatsVO> callback) {
        Call<Result<SalesStatsVO>> call = DehazeSDK.getInstance().getPackageApiService().getSalesStats();
        call.enqueue(callback);
    }
}
