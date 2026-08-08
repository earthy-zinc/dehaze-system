package com.pei.dehaze.ui.system.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.sdk.api.OrderAPI;
import com.pei.dehaze.sdk.model.order.OrderPageVO;
import com.pei.dehaze.sdk.model.order.OrderQuery;
import com.pei.dehaze.sdk.model.order.OrderStatsVO;
import com.pei.dehaze.sdk.model.order.RefundApplyForm;

/**
 * 订单管理 ViewModel — 含列表查询、取消订单、退款处理
 */
public class OrderManageViewModel extends BaseManageViewModel<OrderPageVO> {

    private final MutableLiveData<OrderStatsVO> stats = new MutableLiveData<>();

    @Override
    public void loadData() {
        OrderQuery query = new OrderQuery();
        query.setPageNum(pageNum);
        query.setPageSize(pageSize);
        query.setKeywords(keywords.isEmpty() ? null : keywords);
        OrderAPI.getPage(query, RepositoryAdapters.wrap(withLoading(data -> {
            itemList.postValue(data.getList());
            total.postValue(data.getTotal());
        })));
    }

    public void loadStats(String startTime, String endTime) {
        OrderAPI.getStats(startTime, endTime, RepositoryAdapters.wrap(withLoading(stats::postValue)));
    }

    public LiveData<OrderStatsVO> getStats() { return stats; }

    public void cancelOrder(String orderNo, String reason) {
        OrderAPI.cancel(orderNo, reason, RepositoryAdapters.wrap(withLoading(
                data -> {
                    operationResult.postValue("订单已取消");
                    loadData();
                },
                errorMsg -> {
                    error.postValue(errorMsg);
                    loading.postValue(false);
                }
        )));
    }

    public void applyRefund(String orderNo, String reason) {
        RefundApplyForm form = new RefundApplyForm();
        form.setReason(reason);
        OrderAPI.applyRefund(orderNo, form, RepositoryAdapters.wrap(withLoading(
                data -> {
                    operationResult.postValue("退款申请已提交");
                    loadData();
                },
                errorMsg -> {
                    error.postValue(errorMsg);
                    loading.postValue(false);
                }
        )));
    }
}
