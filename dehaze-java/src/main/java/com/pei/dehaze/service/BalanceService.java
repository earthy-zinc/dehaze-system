package com.pei.dehaze.service;

import com.pei.dehaze.model.form.BalanceRefundAuditForm;
import com.pei.dehaze.model.form.BalanceRefundForm;
import com.pei.dehaze.model.form.RechargeCreateForm;
import com.pei.dehaze.model.vo.BalanceRefundApplyVO;
import com.pei.dehaze.model.vo.BalanceVO;
import com.pei.dehaze.model.vo.RechargeVO;

public interface BalanceService {

    BalanceVO getBalance();

    RechargeVO createRecharge(RechargeCreateForm form);

    BalanceRefundApplyVO applyBalanceRefund(BalanceRefundForm form);

    void approveBalanceRefund(Long refundId, BalanceRefundAuditForm form);

    void payByBalance(Long userId, long amountFen, Long orderId);
}
