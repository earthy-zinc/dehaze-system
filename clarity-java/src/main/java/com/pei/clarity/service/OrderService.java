package com.pei.clarity.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.clarity.model.entity.SysOrder;
import com.pei.clarity.model.form.AutoRenewConfigForm;
import com.pei.clarity.model.form.OrderCreateForm;
import com.pei.clarity.model.form.PayRequest;
import com.pei.clarity.model.form.RefundApplyForm;
import com.pei.clarity.model.form.RefundAuditForm;
import com.pei.clarity.model.query.MyOrderQuery;
import com.pei.clarity.model.query.OrderPageQuery;
import com.pei.clarity.model.query.RefundPageQuery;
import com.pei.clarity.model.vo.AutoRenewConfigVO;
import com.pei.clarity.model.vo.MyOrderVO;
import com.pei.clarity.model.vo.OrderDetailVO;
import com.pei.clarity.model.vo.OrderPageVO;
import com.pei.clarity.model.vo.OrderStatsVO;
import com.pei.clarity.model.vo.PayResult;
import com.pei.clarity.model.vo.RefundRecordVO;
import java.time.LocalDateTime;
import java.util.Map;

public interface OrderService extends IService<SysOrder> {

    PayResult create(OrderCreateForm form);

    PayResult pay(String orderNo, PayRequest request);

    boolean handlePaymentCallback(String channel, Map<String, String> params, String rawBody);

    void cancel(String orderNo, String reason);

    Page<MyOrderVO> listMy(MyOrderQuery query);

    /** 管理端按目标用户查询订单分页（会员消费记录弹窗） */
    Page<MyOrderVO> listByUserId(Long userId, MyOrderQuery query);

    OrderDetailVO getDetail(String orderNo);

    Page<OrderPageVO> getPage(OrderPageQuery query);

    OrderStatsVO getStats(LocalDateTime startTime, LocalDateTime endTime);

    void applyRefund(String orderNo, RefundApplyForm form);

    Page<RefundRecordVO> listRefunds(RefundPageQuery query);

    void approveRefund(Long refundId, RefundAuditForm form);

    void rejectRefund(Long refundId, RefundAuditForm form);

    void updateAutoRenewConfig(AutoRenewConfigForm form);

    AutoRenewConfigVO getAutoRenewConfig(Long packageId);

    void expireOrders();

    void executeRenewal();

    void completeExpiredOrders();

    void expireUserCoupons();

    void retryFailedRefunds();
}
