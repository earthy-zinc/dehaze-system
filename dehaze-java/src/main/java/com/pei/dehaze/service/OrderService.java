package com.pei.dehaze.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.dehaze.model.entity.SysOrder;
import com.pei.dehaze.model.form.AutoRenewConfigForm;
import com.pei.dehaze.model.form.OrderCreateForm;
import com.pei.dehaze.model.form.PayRequest;
import com.pei.dehaze.model.form.RefundApplyForm;
import com.pei.dehaze.model.form.RefundAuditForm;
import com.pei.dehaze.model.query.MyOrderQuery;
import com.pei.dehaze.model.query.OrderPageQuery;
import com.pei.dehaze.model.query.RefundPageQuery;
import com.pei.dehaze.model.vo.AutoRenewConfigVO;
import com.pei.dehaze.model.vo.MyOrderVO;
import com.pei.dehaze.model.vo.OrderDetailVO;
import com.pei.dehaze.model.vo.OrderPageVO;
import com.pei.dehaze.model.vo.OrderStatsVO;
import com.pei.dehaze.model.vo.PayResult;
import com.pei.dehaze.model.vo.RefundRecordVO;

import java.time.LocalDateTime;

public interface OrderService extends IService<SysOrder> {

    PayResult create(OrderCreateForm form);

    PayResult pay(String orderNo, PayRequest request);

    void cancel(String orderNo, String reason);

    Page<MyOrderVO> listMy(MyOrderQuery query);

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
}
