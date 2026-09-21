package com.pei.dehaze.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.SysBalanceLogMapper;
import com.pei.dehaze.mapper.SysBalanceMapper;
import com.pei.dehaze.mapper.SysBalanceRefundMapper;
import com.pei.dehaze.mapper.SysRechargeMapper;
import com.pei.dehaze.model.entity.SysBalance;
import com.pei.dehaze.model.entity.SysBalanceLog;
import com.pei.dehaze.model.entity.SysBalanceRefund;
import com.pei.dehaze.model.entity.SysRecharge;
import com.pei.dehaze.model.form.BalanceRefundAuditForm;
import com.pei.dehaze.model.form.BalanceRefundForm;
import com.pei.dehaze.model.form.RechargeCreateForm;
import com.pei.dehaze.model.vo.BalanceRefundApplyVO;
import com.pei.dehaze.model.vo.BalanceVO;
import com.pei.dehaze.model.vo.RechargeVO;
import com.pei.dehaze.security.util.SecurityUtils;
import com.pei.dehaze.service.BalanceService;
import com.pei.dehaze.service.payment.PaymentChannelService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;

@Slf4j
@Service
@RequiredArgsConstructor
public class BalanceServiceImpl implements BalanceService {

    private static final List<String> RECHARGE_PAY_METHODS = List.of("wechat", "alipay");

    private final SysBalanceMapper balanceMapper;
    private final SysBalanceLogMapper balanceLogMapper;
    private final SysRechargeMapper rechargeMapper;
    private final SysBalanceRefundMapper balanceRefundMapper;
    private final List<PaymentChannelService> paymentChannelServices;

    @Override
    public BalanceVO getBalance() {
        SysBalance account = getOrCreateAccount(SecurityUtils.getUserId());
        BalanceVO vo = new BalanceVO();
        vo.setBalance(account.getBalance());
        vo.setFrozenBalance(account.getFrozenBalance());
        return vo;
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public RechargeVO createRecharge(RechargeCreateForm form) {
        if (!RECHARGE_PAY_METHODS.contains(form.getPayMethod())) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "充值仅支持微信/支付宝支付");
        }
        Long userId = SecurityUtils.getUserId();
        String rechargeNo = "RC" + LocalDateTime.now().format(java.time.format.DateTimeFormatter.ofPattern("yyyyMMddHHmmss"))
                + ThreadLocalRandom.current().nextInt(100000, 999999);
        PaymentChannelService channel = getPaymentChannel(form.getPayMethod());
        PaymentChannelService.UnifiedOrderResult orderResult =
                channel.unifiedOrder(rechargeNo, form.getAmount(), "余额充值", Map.of());
        if (!orderResult.success()) {
            throw new BusinessException(ResultCode.CALL_THIRD_PARTY_SERVICE_ERROR, orderResult.errorMessage());
        }
        SysRecharge recharge = new SysRecharge();
        recharge.setRechargeNo(rechargeNo);
        recharge.setUserId(userId);
        recharge.setAmount(form.getAmount());
        recharge.setPayMethod(form.getPayMethod());
        recharge.setStatus(1);
        rechargeMapper.insert(recharge);

        RechargeVO vo = new RechargeVO();
        vo.setRechargeNo(rechargeNo);
        vo.setPayMethod(form.getPayMethod());
        vo.setPayUrl(orderResult.payUrl());
        vo.setQrCode(orderResult.qrCode());
        return vo;
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public BalanceRefundApplyVO applyBalanceRefund(BalanceRefundForm form) {
        Long userId = SecurityUtils.getUserId();
        SysBalance account = getOrCreateAccount(userId);
        long amount = form.getAmount() != null && form.getAmount() > 0
                ? form.getAmount() : account.getBalance();
        if (amount > account.getBalance()) {
            throw new BusinessException(ResultCode.BALANCE_INSUFFICIENT);
        }
        SysBalanceRefund pending = balanceRefundMapper.selectOne(new LambdaQueryWrapper<SysBalanceRefund>()
                .eq(SysBalanceRefund::getUserId, userId)
                .eq(SysBalanceRefund::getStatus, 1)
                .last("LIMIT 1"));
        if (pending != null) {
            throw new BusinessException(ResultCode.OPERATION_NOT_ALLOW, "已存在待审核的余额退款申请");
        }
        SysBalanceRefund refund = new SysBalanceRefund();
        refund.setRefundNo("BR" + LocalDateTime.now().format(java.time.format.DateTimeFormatter.ofPattern("yyyyMMddHHmmss"))
                + ThreadLocalRandom.current().nextInt(100, 999));
        refund.setUserId(userId);
        refund.setAmount(amount);
        refund.setStatus(1);
        refund.setApplyTime(LocalDateTime.now());
        balanceRefundMapper.insert(refund);

        BalanceRefundApplyVO vo = new BalanceRefundApplyVO();
        vo.setRefundNo(refund.getRefundNo());
        vo.setAmount(amount);
        return vo;
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void approveBalanceRefund(Long refundId, BalanceRefundAuditForm form) {
        SysBalanceRefund refund = balanceRefundMapper.selectById(refundId);
        if (refund == null) {
            throw new BusinessException(ResultCode.REFUND_NOT_FOUND);
        }
        if (refund.getStatus() != 1) {
            throw new BusinessException(ResultCode.ORDER_STATUS_INVALID);
        }
        SysBalance account = getOrCreateAccount(refund.getUserId());
        if (account.getBalance() < refund.getAmount()) {
            throw new BusinessException(ResultCode.BALANCE_INSUFFICIENT);
        }
        if (account.getFrozenBalance() != null && account.getFrozenBalance() > 0) {
            throw new BusinessException(ResultCode.ORDER_STATUS_INVALID, "存在冻结余额，暂不可退");
        }

        Long auditorId = SecurityUtils.getUserId();
        LocalDateTime now = LocalDateTime.now();
        refund.setAuditTime(now);
        refund.setAuditorId(auditorId);
        refund.setAuditRemark(form.getRemark() != null ? form.getRemark() : "");
        try {
            String channelRefundNo = refundViaChannel(refund, form.getChannel());
            refund.setChannel(form.getChannel());
            refund.setChannelRefundNo(channelRefundNo);
            withdraw(refund.getUserId(), refund.getAmount(), refund.getId());
            refund.setStatus(2);
            refund.setRefundTime(now);
        } catch (Exception e) {
            log.error("余额退款执行失败 refundId={}: {}", refundId, e.getMessage(), e);
            refund.setStatus(3);
            refund.setErrorMessage(e.getMessage());
        }
        balanceRefundMapper.updateById(refund);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void payByBalance(Long userId, long amountFen, Long orderId) {
        getOrCreateAccount(userId);
        int rows = balanceMapper.deductBalance(userId, amountFen);
        if (rows == 0) {
            throw new BusinessException(ResultCode.BALANCE_INSUFFICIENT);
        }
        SysBalance account = getOrCreateAccount(userId);
        createBalanceLog(userId, "consume", -amountFen, account.getBalance(), orderId);
    }

    private String refundViaChannel(SysBalanceRefund refund, String channel) {
        if (channel == null || !RECHARGE_PAY_METHODS.contains(channel)) {
            return null;
        }
        PaymentChannelService paymentChannel = getPaymentChannel(channel);
        boolean ok = paymentChannel.refund(refund.getRefundNo(), refund.getRefundNo(),
                refund.getAmount(), refund.getAmount(), "余额退款");
        if (!ok) {
            throw new BusinessException(ResultCode.CALL_THIRD_PARTY_SERVICE_ERROR, "渠道退款失败");
        }
        return null;
    }

    private void withdraw(Long userId, long amount, Long relatedId) {
        int rows = balanceMapper.deductBalance(userId, amount);
        if (rows == 0) {
            throw new BusinessException(ResultCode.BALANCE_INSUFFICIENT);
        }
        SysBalance account = getOrCreateAccount(userId);
        createBalanceLog(userId, "refund", -amount, account.getBalance(), relatedId);
    }

    private SysBalance getOrCreateAccount(Long userId) {
        SysBalance account = balanceMapper.selectOne(new LambdaQueryWrapper<SysBalance>()
                .eq(SysBalance::getUserId, userId));
        if (account != null) {
            return account;
        }
        account = new SysBalance();
        account.setUserId(userId);
        account.setBalance(0L);
        account.setFrozenBalance(0L);
        account.setVersion(0);
        balanceMapper.insert(account);
        return account;
    }

    private void createBalanceLog(Long userId, String changeType, long amount, long balanceAfter, Long relatedId) {
        SysBalanceLog logRecord = new SysBalanceLog();
        logRecord.setUserId(userId);
        logRecord.setChangeType(changeType);
        logRecord.setAmount(amount);
        logRecord.setBalanceAfter(balanceAfter);
        logRecord.setRelatedId(relatedId);
        balanceLogMapper.insert(logRecord);
    }

    private PaymentChannelService getPaymentChannel(String channelType) {
        return paymentChannelServices.stream()
                .filter(ch -> ch.getChannelType().equals(channelType))
                .findFirst()
                .orElseThrow(() -> new BusinessException(ResultCode.PARAM_ERROR, "不支持的支付渠道: " + channelType));
    }
}
