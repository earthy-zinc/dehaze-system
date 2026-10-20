package com.pei.clarity.service;

import com.pei.clarity.model.form.BalanceRefundAuditForm;
import com.pei.clarity.model.form.BalanceRefundForm;
import com.pei.clarity.model.form.RechargeCreateForm;
import com.pei.clarity.model.vo.BalanceRefundApplyVO;
import com.pei.clarity.model.vo.BalanceVO;
import com.pei.clarity.model.vo.RechargeVO;

public interface BalanceService {

    BalanceVO getBalance();

    RechargeVO createRecharge(RechargeCreateForm form);

    BalanceRefundApplyVO applyBalanceRefund(BalanceRefundForm form);

    void approveBalanceRefund(Long refundId, BalanceRefundAuditForm form);

    void payByBalance(Long userId, long amountFen, Long orderId);
}
