package com.pei.dehaze.service.impl;

import cn.hutool.core.text.CharSequenceUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.SysAutoRenewMapper;
import com.pei.dehaze.mapper.SysOrderMapper;
import com.pei.dehaze.mapper.SysPackageMapper;
import com.pei.dehaze.mapper.SysPaymentRecordMapper;
import com.pei.dehaze.mapper.SysRefundRecordMapper;
import com.pei.dehaze.mapper.SysUserCouponMapper;
import com.pei.dehaze.mapper.SysUserMapper;
import com.pei.dehaze.model.entity.SysAutoRenew;
import com.pei.dehaze.model.entity.SysOrder;
import com.pei.dehaze.model.entity.SysPackage;
import com.pei.dehaze.model.entity.SysPaymentRecord;
import com.pei.dehaze.model.entity.SysRefundRecord;
import com.pei.dehaze.model.entity.SysUser;
import com.pei.dehaze.model.entity.SysUserCoupon;
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
import com.pei.dehaze.model.vo.PaymentRecordVO;
import com.pei.dehaze.model.vo.RefundRecordVO;
import com.pei.dehaze.security.util.SecurityUtils;
import com.pei.dehaze.service.MemberService;
import com.pei.dehaze.service.OrderService;
import com.pei.dehaze.service.PackageService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;

@Slf4j
@Service
@RequiredArgsConstructor
public class OrderServiceImpl extends ServiceImpl<SysOrderMapper, SysOrder> implements OrderService {

    private static final int ORDER_TIMEOUT_MINUTES = 30;
    private static final int MAX_RENEW_FAIL_COUNT = 3;
    private static final DateTimeFormatter ORDER_NO_FORMAT = DateTimeFormatter.ofPattern("yyyyMMddHHmmss");
    private static final DateTimeFormatter DAILY_FORMAT = DateTimeFormatter.ofPattern("yyyy-MM-dd");

    private static final Map<Integer, String> ORDER_STATUS_MAP = Map.of(
            1, "pending", 2, "paid", 3, "completed", 4, "cancelled", 5, "refunding", 6, "refunded");
    private static final Map<String, Integer> ORDER_STATUS_REVERSE_MAP = Map.of(
            "pending", 1, "paid", 2, "completed", 3, "cancelled", 4, "refunding", 5, "refunded", 6);
    private static final Map<Integer, String> REFUND_STATUS_MAP = Map.of(
            1, "refunding", 2, "refunded", 3, "refund_failed");

    private final SysPackageMapper packageMapper;
    private final SysUserMapper userMapper;
    private final SysPaymentRecordMapper paymentRecordMapper;
    private final SysRefundRecordMapper refundRecordMapper;
    private final SysAutoRenewMapper autoRenewMapper;
    private final SysUserCouponMapper userCouponMapper;
    private final PackageService packageService;
    private final MemberService memberService;

    @Override
    @Transactional(rollbackFor = Exception.class)
    public PayResult create(OrderCreateForm form) {
        Long userId = SecurityUtils.getUserId();
        SysPackage pkg = packageMapper.selectById(form.getPackageId());
        if (pkg == null) {
            throw new BusinessException(ResultCode.PACKAGE_NOT_FOUND);
        }
        if (pkg.getStatus() != 1) {
            throw new BusinessException(ResultCode.PACKAGE_OFF_SHELF);
        }
        var priceResult = packageService.calculatePrice(form.getPackageId(), form.getCouponId());
        if (form.getCouponId() != null) {
            lockCoupon(form.getCouponId(), userId);
        }
        LocalDateTime now = LocalDateTime.now();
        SysOrder order = new SysOrder();
        order.setOrderNo(generateOrderNo());
        order.setUserId(userId);
        order.setPackageId(pkg.getId());
        order.setPackageName(pkg.getName());
        order.setPackageLevel(pkg.getLevelCode());
        order.setPeriodDays(pkg.getPeriodDays());
        order.setOriginalPrice(priceResult.getOriginalPrice());
        order.setDiscountAmount(priceResult.getDiscountAmount());
        order.setCouponId(form.getCouponId());
        order.setCouponAmount(priceResult.getCouponAmount());
        order.setPayableAmount(priceResult.getPayableAmount());
        order.setPaidAmount(0L);
        order.setPayMethod(form.getPayMethod());
        order.setStatus(1);
        order.setExpireTime(now.plusMinutes(ORDER_TIMEOUT_MINUTES));
        order.setIsAutoRenew(0);
        this.save(order);

        PayResult result = new PayResult();
        result.setOrderNo(order.getOrderNo());
        result.setPayMethod(form.getPayMethod());
        result.setPaid(false);
        return result;
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public PayResult pay(String orderNo, PayRequest request) {
        SysOrder order = getOrderByNo(orderNo);
        if (order.getStatus() != 1) {
            throw new BusinessException(ResultCode.ORDER_ALREADY_PAID);
        }
        if (order.getExpireTime() != null && order.getExpireTime().isBefore(LocalDateTime.now())) {
            throw new BusinessException(ResultCode.ORDER_EXPIRED);
        }
        order.setPayMethod(request.getPayMethod());
        PayResult result = new PayResult();
        result.setOrderNo(orderNo);
        result.setPayMethod(request.getPayMethod());

        if ("balance".equals(request.getPayMethod())) {
            completePayment(order, request.getPayMethod());
            result.setPaid(true);
        } else {
            SysPaymentRecord record = createPaymentRecord(order, request.getPayMethod(), 1);
            result.setPayUrl("https://mock-pay.example.com/pay?orderNo=" + orderNo + "&method=" + request.getPayMethod());
            result.setQrCode("pay://" + orderNo);
            result.setPaid(false);
        }
        return result;
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void cancel(String orderNo, String reason) {
        SysOrder order = getOrderByNo(orderNo);
        if (order.getStatus() != 1) {
            throw new BusinessException(ResultCode.ORDER_STATUS_INVALID);
        }
        order.setStatus(4);
        order.setCancelReason(reason);
        this.updateById(order);
        if (order.getCouponId() != null) {
            unlockCoupon(order.getCouponId());
        }
    }

    @Override
    public Page<MyOrderVO> listMy(MyOrderQuery query) {
        Long userId = SecurityUtils.getUserId();
        Page<SysOrder> page = new Page<>(query.getPageNum(), query.getPageSize());
        LambdaQueryWrapper<SysOrder> wrapper = new LambdaQueryWrapper<SysOrder>()
                .eq(SysOrder::getUserId, userId)
                .eq(CharSequenceUtil.isNotBlank(query.getStatus()), SysOrder::getStatus, orderStatusToInt(query.getStatus()))
                .orderByDesc(SysOrder::getId);
        this.page(page, wrapper);
        Page<MyOrderVO> result = new Page<>(page.getCurrent(), page.getSize(), page.getTotal());
        result.setRecords(page.getRecords().stream().map(this::toMyOrderVO).toList());
        return result;
    }

    @Override
    public OrderDetailVO getDetail(String orderNo) {
        SysOrder order = getOrderByNo(orderNo);
        OrderDetailVO vo = new OrderDetailVO();
        copyOrderPageFields(order, vo);
        vo.setExpireTime(order.getExpireTime());
        vo.setEffectiveTime(order.getEffectiveTime());
        vo.setCancelReason(order.getCancelReason());
        vo.setIsAutoRenew(order.getIsAutoRenew());
        List<SysPaymentRecord> payments = paymentRecordMapper.selectList(new LambdaQueryWrapper<SysPaymentRecord>()
                .eq(SysPaymentRecord::getOrderId, order.getId())
                .orderByDesc(SysPaymentRecord::getId));
        vo.setPaymentRecords(payments.stream().map(this::toPaymentRecordVO).toList());
        SysRefundRecord refund = refundRecordMapper.selectOne(new LambdaQueryWrapper<SysRefundRecord>()
                .eq(SysRefundRecord::getOrderId, order.getId())
                .orderByDesc(SysRefundRecord::getId)
                .last("LIMIT 1"));
        vo.setRefundRecord(refund != null ? toRefundRecordVO(refund) : null);
        return vo;
    }

    @Override
    public Page<OrderPageVO> getPage(OrderPageQuery query) {
        Page<SysOrder> page = new Page<>(query.getPageNum(), query.getPageSize());
        LambdaQueryWrapper<SysOrder> wrapper = new LambdaQueryWrapper<SysOrder>()
                .eq(CharSequenceUtil.isNotBlank(query.getOrderNo()), SysOrder::getOrderNo, query.getOrderNo())
                .eq(CharSequenceUtil.isNotBlank(query.getStatus()), SysOrder::getStatus, orderStatusToInt(query.getStatus()))
                .eq(CharSequenceUtil.isNotBlank(query.getPayMethod()), SysOrder::getPayMethod, query.getPayMethod())
                .ge(query.getAmountMin() != null, SysOrder::getPayableAmount, query.getAmountMin())
                .le(query.getAmountMax() != null, SysOrder::getPayableAmount, query.getAmountMax())
                .ge(query.getPaidTimeStart() != null, SysOrder::getPaidTime, query.getPaidTimeStart())
                .le(query.getPaidTimeEnd() != null, SysOrder::getPaidTime, query.getPaidTimeEnd())
                .orderByDesc(SysOrder::getId);

        if (CharSequenceUtil.isNotBlank(query.getKeywords())) {
            List<Long> userIds = userMapper.selectList(new LambdaQueryWrapper<SysUser>()
                            .and(w -> w.like(SysUser::getUsername, query.getKeywords())
                                    .or().like(SysUser::getNickname, query.getKeywords())))
                    .stream().map(SysUser::getId).toList();
            if (userIds.isEmpty()) {
                Page<OrderPageVO> empty = new Page<>(page.getCurrent(), page.getSize(), 0);
                empty.setRecords(Collections.emptyList());
                return empty;
            }
            wrapper.in(SysOrder::getUserId, userIds);
        }

        this.page(page, wrapper);
        Map<Long, SysUser> userMap = userMapper.selectBatchIds(page.getRecords().stream()
                        .map(SysOrder::getUserId).distinct().toList())
                .stream().collect(Collectors.toMap(SysUser::getId, u -> u));

        Page<OrderPageVO> result = new Page<>(page.getCurrent(), page.getSize(), page.getTotal());
        result.setRecords(page.getRecords().stream().map(o -> toOrderPageVO(o, userMap.get(o.getUserId()))).toList());
        return result;
    }

    @Override
    public OrderStatsVO getStats(LocalDateTime startTime, LocalDateTime endTime) {
        LambdaQueryWrapper<SysOrder> wrapper = new LambdaQueryWrapper<SysOrder>()
                .ge(startTime != null, SysOrder::getCreateTime, startTime)
                .le(endTime != null, SysOrder::getCreateTime, endTime);
        List<SysOrder> orders = this.list(wrapper);

        OrderStatsVO stats = new OrderStatsVO();
        stats.setTotalOrders((long) orders.size());

        List<SysOrder> paidOrders = orders.stream().filter(o -> Arrays.asList(2, 3, 5, 6).contains(o.getStatus())).toList();
        stats.setTotalRevenue(paidOrders.stream().mapToLong(o -> o.getPaidAmount() != null ? o.getPaidAmount() : 0).sum());

        List<SysOrder> refundedOrders = orders.stream().filter(o -> o.getStatus() == 6).toList();
        stats.setTotalRefund(refundedOrders.stream().mapToLong(o -> o.getPaidAmount() != null ? o.getPaidAmount() : 0).sum());
        stats.setRefundRate(stats.getTotalRevenue() > 0 ? (double) stats.getTotalRefund() / stats.getTotalRevenue() : 0.0);

        Map<String, Long> statusDist = new LinkedHashMap<>();
        for (String s : Arrays.asList("pending", "paid", "completed", "cancelled", "refunding", "refunded")) {
            statusDist.put(s, 0L);
        }
        for (SysOrder o : orders) {
            String status = ORDER_STATUS_MAP.get(o.getStatus());
            if (status != null) {
                statusDist.merge(status, 1L, Long::sum);
            }
        }
        stats.setStatusDistribution(statusDist);

        Map<String, Long> payMethodDist = new LinkedHashMap<>();
        for (String m : Arrays.asList("wechat", "alipay", "balance", "combined")) {
            payMethodDist.put(m, 0L);
        }
        for (SysOrder o : paidOrders) {
            if (o.getPayMethod() != null && payMethodDist.containsKey(o.getPayMethod())) {
                payMethodDist.merge(o.getPayMethod(), 1L, Long::sum);
            }
        }
        stats.setPayMethodDistribution(payMethodDist);

        Map<Long, OrderStatsVO.PackageStatItem> pkgStatsMap = new LinkedHashMap<>();
        for (SysOrder o : paidOrders) {
            pkgStatsMap.computeIfAbsent(o.getPackageId(), k -> {
                OrderStatsVO.PackageStatItem item = new OrderStatsVO.PackageStatItem();
                item.setPackageId(o.getPackageId());
                item.setPackageName(o.getPackageName());
                item.setCount(0L);
                item.setRevenue(0L);
                return item;
            });
            pkgStatsMap.get(o.getPackageId()).setCount(pkgStatsMap.get(o.getPackageId()).getCount() + 1);
            pkgStatsMap.get(o.getPackageId()).setRevenue(pkgStatsMap.get(o.getPackageId()).getRevenue() + (o.getPaidAmount() != null ? o.getPaidAmount() : 0));
        }
        stats.setPackageDistribution(new ArrayList<>(pkgStatsMap.values()));

        Map<String, OrderStatsVO.DailyStatItem> dailyMap = new LinkedHashMap<>();
        for (SysOrder o : paidOrders) {
            if (o.getCreateTime() == null) continue;
            String date = o.getCreateTime().format(DAILY_FORMAT);
            dailyMap.computeIfAbsent(date, d -> {
                OrderStatsVO.DailyStatItem item = new OrderStatsVO.DailyStatItem();
                item.setDate(d);
                item.setCount(0L);
                item.setRevenue(0L);
                return item;
            });
            dailyMap.get(date).setCount(dailyMap.get(date).getCount() + 1);
            dailyMap.get(date).setRevenue(dailyMap.get(date).getRevenue() + (o.getPaidAmount() != null ? o.getPaidAmount() : 0));
        }
        stats.setDailyStats(new ArrayList<>(dailyMap.values()));
        return stats;
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void applyRefund(String orderNo, RefundApplyForm form) {
        SysOrder order = getOrderByNo(orderNo);
        if (order.getStatus() != 2 && order.getStatus() != 3) {
            throw new BusinessException(ResultCode.ORDER_STATUS_INVALID);
        }
        SysRefundRecord refund = new SysRefundRecord();
        refund.setRefundNo("RF" + generateOrderNo());
        refund.setOrderId(order.getId());
        refund.setUserId(order.getUserId());
        refund.setRefundAmount(order.getPaidAmount());
        String reason = form.getReason();
        if (CharSequenceUtil.isNotBlank(form.getCustomReason())) {
            reason = reason + ":" + form.getCustomReason();
        }
        refund.setReason(reason);
        refund.setUsedQuota(0);
        refund.setStatus(1);
        refund.setChannel(order.getPayMethod());
        refund.setApplyTime(LocalDateTime.now());
        refundRecordMapper.insert(refund);

        order.setStatus(5);
        this.updateById(order);
    }

    @Override
    public Page<RefundRecordVO> listRefunds(RefundPageQuery query) {
        LambdaQueryWrapper<SysRefundRecord> wrapper = new LambdaQueryWrapper<SysRefundRecord>()
                .eq(CharSequenceUtil.isNotBlank(query.getStatus()), SysRefundRecord::getStatus, refundStatusToInt(query.getStatus()))
                .ge(query.getApplyTimeStart() != null, SysRefundRecord::getApplyTime, query.getApplyTimeStart())
                .le(query.getApplyTimeEnd() != null, SysRefundRecord::getApplyTime, query.getApplyTimeEnd())
                .orderByDesc(SysRefundRecord::getId);

        List<SysRefundRecord> refunds = refundRecordMapper.selectList(wrapper);
        Map<Long, SysOrder> orderMap = new HashMap<>();
        Map<Long, SysUser> userMap = new HashMap<>();
        if (!refunds.isEmpty()) {
            List<Long> orderIds = refunds.stream().map(SysRefundRecord::getOrderId).distinct().toList();
            List<SysOrder> orders = this.listByIds(orderIds);
            orderMap.putAll(orders.stream().collect(Collectors.toMap(SysOrder::getId, o -> o)));
            List<Long> userIds = refunds.stream().map(SysRefundRecord::getUserId).distinct().toList();
            userMap.putAll(userMapper.selectBatchIds(userIds).stream().collect(Collectors.toMap(SysUser::getId, u -> u)));
        }

        String orderNoFilter = query.getOrderNo();
        if (CharSequenceUtil.isNotBlank(orderNoFilter)) {
            refunds = refunds.stream().filter(r -> {
                SysOrder o = orderMap.get(r.getOrderId());
                return o != null && o.getOrderNo().contains(orderNoFilter);
            }).toList();
        }
        String keywordsFilter = query.getKeywords();
        if (CharSequenceUtil.isNotBlank(keywordsFilter)) {
            refunds = refunds.stream().filter(r -> {
                SysUser u = userMap.get(r.getUserId());
                return u != null && (u.getUsername().contains(keywordsFilter) || u.getNickname().contains(keywordsFilter));
            }).toList();
        }

        long total = refunds.size();
        int fromIndex = (int) Math.min((query.getPageNum() - 1) * query.getPageSize(), total);
        int toIndex = (int) Math.min(fromIndex + query.getPageSize(), total);
        List<SysRefundRecord> pageRecords = refunds.subList(fromIndex, toIndex);

        Page<RefundRecordVO> result = new Page<>(query.getPageNum(), query.getPageSize(), total);
        result.setRecords(pageRecords.stream().map(r -> toRefundRecordVO(r, orderMap.get(r.getOrderId()), userMap.get(r.getUserId()))).toList());
        return result;
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void approveRefund(Long refundId, RefundAuditForm form) {
        SysRefundRecord refund = refundRecordMapper.selectById(refundId);
        if (refund == null) {
            throw new BusinessException(ResultCode.REFUND_NOT_FOUND);
        }
        if (refund.getStatus() != 1) {
            throw new BusinessException(ResultCode.ORDER_STATUS_INVALID);
        }
        Long operatorId = SecurityUtils.getUserId();
        refund.setStatus(2);
        refund.setAuditTime(LocalDateTime.now());
        refund.setAuditorId(operatorId);
        refund.setAuditRemark(form.getRemark());
        refund.setRefundTime(LocalDateTime.now());
        refundRecordMapper.updateById(refund);

        SysOrder order = this.getById(refund.getOrderId());
        if (order != null) {
            order.setStatus(6);
            this.updateById(order);
            if (order.getCouponId() != null) {
                unlockCoupon(order.getCouponId());
            }
        }
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void rejectRefund(Long refundId, RefundAuditForm form) {
        SysRefundRecord refund = refundRecordMapper.selectById(refundId);
        if (refund == null) {
            throw new BusinessException(ResultCode.REFUND_NOT_FOUND);
        }
        if (refund.getStatus() != 1) {
            throw new BusinessException(ResultCode.ORDER_STATUS_INVALID);
        }
        Long operatorId = SecurityUtils.getUserId();
        refund.setStatus(3);
        refund.setAuditTime(LocalDateTime.now());
        refund.setAuditorId(operatorId);
        refund.setAuditRemark(form.getRemark());
        refundRecordMapper.updateById(refund);

        SysOrder order = this.getById(refund.getOrderId());
        if (order != null && order.getStatus() == 5) {
            order.setStatus(2);
            this.updateById(order);
        }
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void updateAutoRenewConfig(AutoRenewConfigForm form) {
        Long userId = SecurityUtils.getUserId();
        SysPackage pkg = packageMapper.selectById(form.getPackageId());
        if (pkg == null) {
            throw new BusinessException(ResultCode.PACKAGE_NOT_FOUND);
        }
        SysAutoRenew autoRenew = autoRenewMapper.selectOne(new LambdaQueryWrapper<SysAutoRenew>()
                .eq(SysAutoRenew::getUserId, userId)
                .eq(SysAutoRenew::getPackageId, form.getPackageId()));
        if (autoRenew == null) {
            autoRenew = new SysAutoRenew();
            autoRenew.setUserId(userId);
            autoRenew.setPackageId(form.getPackageId());
            autoRenew.setPayMethod(form.getPayMethod());
            autoRenew.setFailCount(0);
        } else {
            autoRenew.setPayMethod(form.getPayMethod());
        }
        if (form.getEnabled()) {
            autoRenew.setStatus(1);
            autoRenew.setCloseReason(null);
            autoRenew.setNextRenewTime(LocalDateTime.now().plusDays(pkg.getPeriodDays()));
        } else {
            autoRenew.setStatus(0);
            autoRenew.setCloseReason("用户手动关闭");
            autoRenew.setNextRenewTime(null);
        }
        if (autoRenew.getId() == null) {
            autoRenewMapper.insert(autoRenew);
        } else {
            autoRenewMapper.updateById(autoRenew);
        }
    }

    @Override
    public AutoRenewConfigVO getAutoRenewConfig(Long packageId) {
        Long userId = SecurityUtils.getUserId();
        SysPackage pkg = packageMapper.selectById(packageId);
        if (pkg == null) {
            throw new BusinessException(ResultCode.PACKAGE_NOT_FOUND);
        }
        SysAutoRenew autoRenew = autoRenewMapper.selectOne(new LambdaQueryWrapper<SysAutoRenew>()
                .eq(SysAutoRenew::getUserId, userId)
                .eq(SysAutoRenew::getPackageId, packageId));
        AutoRenewConfigVO vo = new AutoRenewConfigVO();
        vo.setUserId(userId);
        vo.setPackageId(packageId);
        vo.setPackageName(pkg.getName());
        if (autoRenew != null) {
            vo.setPayMethod(autoRenew.getPayMethod());
            vo.setEnabled(autoRenew.getStatus() != null && autoRenew.getStatus() == 1);
            vo.setNextRenewTime(autoRenew.getNextRenewTime());
            vo.setFailCount(autoRenew.getFailCount() != null ? autoRenew.getFailCount() : 0);
            vo.setCloseReason(autoRenew.getCloseReason());
        } else {
            vo.setPayMethod("balance");
            vo.setEnabled(false);
            vo.setFailCount(0);
        }
        return vo;
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void expireOrders() {
        List<SysOrder> pendingOrders = this.list(new LambdaQueryWrapper<SysOrder>()
                .eq(SysOrder::getStatus, 1)
                .lt(SysOrder::getExpireTime, LocalDateTime.now()));
        for (SysOrder order : pendingOrders) {
            order.setStatus(4);
            order.setCancelReason("系统超时自动取消");
            this.updateById(order);
            if (order.getCouponId() != null) {
                unlockCoupon(order.getCouponId());
            }
        }
        log.info("订单超时取消: 共处理{}条", pendingOrders.size());
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void executeRenewal() {
        List<SysAutoRenew> renewals = autoRenewMapper.selectList(new LambdaQueryWrapper<SysAutoRenew>()
                .eq(SysAutoRenew::getStatus, 1)
                .le(SysAutoRenew::getNextRenewTime, LocalDateTime.now()));
        for (SysAutoRenew renewal : renewals) {
            try {
                executeSingleRenewal(renewal);
            } catch (Exception e) {
                log.error("自动续费失败: userId={}, packageId={}", renewal.getUserId(), renewal.getPackageId(), e);
                renewal.setFailCount((renewal.getFailCount() != null ? renewal.getFailCount() : 0) + 1);
                if (renewal.getFailCount() >= MAX_RENEW_FAIL_COUNT) {
                    renewal.setStatus(0);
                    renewal.setCloseReason("连续失败超过限制");
                }
                autoRenewMapper.updateById(renewal);
            }
        }
        log.info("自动续费执行完成: 共处理{}条", renewals.size());
    }

    private void executeSingleRenewal(SysAutoRenew renewal) {
        SysPackage pkg = packageMapper.selectById(renewal.getPackageId());
        if (pkg == null || pkg.getStatus() != 1) {
            throw new BusinessException(ResultCode.PACKAGE_NOT_FOUND);
        }
        SysOrder order = new SysOrder();
        order.setOrderNo(generateOrderNo());
        order.setUserId(renewal.getUserId());
        order.setPackageId(pkg.getId());
        order.setPackageName(pkg.getName());
        order.setPackageLevel(pkg.getLevelCode());
        order.setPeriodDays(pkg.getPeriodDays());
        order.setOriginalPrice(pkg.getOriginalPrice());
        order.setDiscountAmount(0L);
        order.setCouponAmount(0L);
        order.setPayableAmount(pkg.getSalePrice());
        order.setPaidAmount(pkg.getSalePrice());
        order.setPayMethod(renewal.getPayMethod());
        order.setStatus(2);
        order.setPaidTime(LocalDateTime.now());
        order.setEffectiveTime(LocalDateTime.now());
        order.setPackageExpireTime(LocalDateTime.now().plusDays(pkg.getPeriodDays()));
        order.setIsAutoRenew(1);
        this.save(order);

        createPaymentRecord(order, renewal.getPayMethod(), 2);
        updatePackageSalesCount(pkg.getId());
        updateMemberOnPayment(renewal.getUserId(), pkg);

        renewal.setLastRenewOrderId(order.getId());
        renewal.setNextRenewTime(order.getPackageExpireTime());
        renewal.setFailCount(0);
        autoRenewMapper.updateById(renewal);
    }

    private void completePayment(SysOrder order, String payMethod) {
        LocalDateTime now = LocalDateTime.now();
        order.setStatus(2);
        order.setPaidTime(now);
        order.setEffectiveTime(now);
        order.setPaidAmount(order.getPayableAmount());
        order.setPackageExpireTime(now.plusDays(order.getPeriodDays() != null ? order.getPeriodDays() : 30));
        this.updateById(order);

        createPaymentRecord(order, payMethod, 2);
        updatePackageSalesCount(order.getPackageId());
        updateMemberOnPayment(order.getUserId(), order.getPackageId());
        if (order.getCouponId() != null) {
            consumeCoupon(order.getCouponId(), order.getId());
        }
    }

    private SysPaymentRecord createPaymentRecord(SysOrder order, String channel, int status) {
        SysPaymentRecord record = new SysPaymentRecord();
        record.setOrderId(order.getId());
        record.setUserId(order.getUserId());
        record.setPaymentNo("PAY" + System.currentTimeMillis() + ThreadLocalRandom.current().nextInt(1000, 9999));
        record.setChannel(channel);
        record.setAmount(order.getPayableAmount());
        record.setStatus(status);
        if (status == 2) {
            record.setCallbackTime(LocalDateTime.now());
        }
        paymentRecordMapper.insert(record);
        return record;
    }

    private void updatePackageSalesCount(Long packageId) {
        SysPackage pkg = packageMapper.selectById(packageId);
        if (pkg != null) {
            LambdaUpdateWrapper<SysPackage> wrapper = new LambdaUpdateWrapper<SysPackage>()
                    .eq(SysPackage::getId, packageId)
                    .set(SysPackage::getSalesCount, (pkg.getSalesCount() != null ? pkg.getSalesCount() : 0) + 1);
            packageMapper.update(null, wrapper);
        }
    }

    private void updateMemberOnPayment(Long userId, Long packageId) {
        SysPackage pkg = packageMapper.selectById(packageId);
        if (pkg != null) {
            updateMemberOnPayment(userId, pkg);
        }
    }

    private void updateMemberOnPayment(Long userId, SysPackage pkg) {
        try {
            var form = new com.pei.dehaze.model.form.MemberLevelAdjustForm();
            form.setLevelCode(pkg.getLevelCode());
            form.setExpireTime(LocalDateTime.now().plusDays(pkg.getPeriodDays()));
            form.setReason("套餐订单支付: " + pkg.getName());
            memberService.adjustLevel(userId, form);
        } catch (Exception e) {
            log.warn("会员等级激活失败: userId={}, packageId={}", userId, pkg.getId(), e);
        }
    }

    private void lockCoupon(Long userCouponId, Long userId) {
        SysUserCoupon userCoupon = userCouponMapper.selectById(userCouponId);
        if (userCoupon == null) {
            throw new BusinessException(ResultCode.COUPON_NOT_FOUND);
        }
        if (!userCoupon.getUserId().equals(userId)) {
            throw new BusinessException(ResultCode.COUPON_NOT_FOUND);
        }
        if (userCoupon.getStatus() != 1) {
            throw new BusinessException(ResultCode.COUPON_ALREADY_USED);
        }
        userCoupon.setStatus(4);
        userCouponMapper.updateById(userCoupon);
    }

    private void unlockCoupon(Long userCouponId) {
        SysUserCoupon userCoupon = userCouponMapper.selectById(userCouponId);
        if (userCoupon != null && userCoupon.getStatus() == 4) {
            userCoupon.setStatus(1);
            userCouponMapper.updateById(userCoupon);
        }
    }

    private void consumeCoupon(Long userCouponId, Long orderId) {
        SysUserCoupon userCoupon = userCouponMapper.selectById(userCouponId);
        if (userCoupon != null && userCoupon.getStatus() == 4) {
            userCoupon.setStatus(2);
            userCoupon.setUsedTime(LocalDateTime.now());
            userCoupon.setUsedOrderId(orderId);
            userCouponMapper.updateById(userCoupon);
        }
    }

    private SysOrder getOrderByNo(String orderNo) {
        SysOrder order = this.getOne(new LambdaQueryWrapper<SysOrder>()
                .eq(SysOrder::getOrderNo, orderNo));
        if (order == null) {
            throw new BusinessException(ResultCode.ORDER_NOT_FOUND);
        }
        return order;
    }

    private String generateOrderNo() {
        return LocalDateTime.now().format(ORDER_NO_FORMAT) + ThreadLocalRandom.current().nextInt(100000, 999999);
    }

    private Integer orderStatusToInt(String status) {
        if (CharSequenceUtil.isBlank(status)) {
            return null;
        }
        return ORDER_STATUS_REVERSE_MAP.get(status);
    }

    private Integer refundStatusToInt(String status) {
        if (CharSequenceUtil.isBlank(status)) {
            return null;
        }
        return switch (status) {
            case "refunding" -> 1;
            case "refunded" -> 2;
            case "refund_failed" -> 3;
            default -> null;
        };
    }

    private MyOrderVO toMyOrderVO(SysOrder order) {
        MyOrderVO vo = new MyOrderVO();
        vo.setId(order.getId());
        vo.setOrderNo(order.getOrderNo());
        vo.setPackageName(order.getPackageName());
        vo.setPackageLevel(order.getPackageLevel());
        vo.setPayableAmount(order.getPayableAmount());
        vo.setPaidAmount(order.getPaidAmount());
        vo.setPayMethod(order.getPayMethod());
        vo.setStatus(ORDER_STATUS_MAP.get(order.getStatus()));
        vo.setCreateTime(order.getCreateTime());
        vo.setPaidTime(order.getPaidTime());
        vo.setPackageExpireTime(order.getPackageExpireTime());
        return vo;
    }

    private OrderPageVO toOrderPageVO(SysOrder order, SysUser user) {
        OrderPageVO vo = new OrderPageVO();
        copyOrderPageFields(order, vo);
        vo.setUserId(order.getUserId());
        vo.setUsername(user != null ? user.getUsername() : null);
        vo.setOriginalPrice(order.getOriginalPrice());
        vo.setDiscountAmount(order.getDiscountAmount());
        vo.setCouponAmount(order.getCouponAmount());
        return vo;
    }

    private void copyOrderPageFields(SysOrder order, OrderPageVO vo) {
        vo.setId(order.getId());
        vo.setOrderNo(order.getOrderNo());
        vo.setPackageName(order.getPackageName());
        vo.setPackageLevel(order.getPackageLevel());
        vo.setPayableAmount(order.getPayableAmount());
        vo.setPaidAmount(order.getPaidAmount());
        vo.setPayMethod(order.getPayMethod());
        vo.setStatus(ORDER_STATUS_MAP.get(order.getStatus()));
        vo.setCreateTime(order.getCreateTime());
        vo.setPaidTime(order.getPaidTime());
        vo.setPackageExpireTime(order.getPackageExpireTime());
    }

    private PaymentRecordVO toPaymentRecordVO(SysPaymentRecord record) {
        PaymentRecordVO vo = new PaymentRecordVO();
        vo.setId(record.getId());
        vo.setPaymentNo(record.getPaymentNo());
        vo.setChannel(record.getChannel());
        vo.setAmount(record.getAmount());
        vo.setStatus(record.getStatus());
        vo.setCallbackTime(record.getCallbackTime());
        vo.setCreateTime(record.getCreateTime());
        return vo;
    }

    private RefundRecordVO toRefundRecordVO(SysRefundRecord refund) {
        return toRefundRecordVO(refund, null, null);
    }

    private RefundRecordVO toRefundRecordVO(SysRefundRecord refund, SysOrder order, SysUser user) {
        RefundRecordVO vo = new RefundRecordVO();
        vo.setId(refund.getId());
        vo.setRefundNo(refund.getRefundNo());
        vo.setOrderId(refund.getOrderId());
        vo.setOrderNo(order != null ? order.getOrderNo() : null);
        vo.setUserId(refund.getUserId());
        vo.setUsername(user != null ? user.getUsername() : null);
        vo.setRefundAmount(refund.getRefundAmount());
        vo.setReason(refund.getReason());
        vo.setUsedQuota(refund.getUsedQuota());
        vo.setStatus(REFUND_STATUS_MAP.get(refund.getStatus()));
        vo.setChannel(refund.getChannel());
        vo.setChannelRefundNo(refund.getChannelRefundNo());
        vo.setApplyTime(refund.getApplyTime());
        vo.setAuditTime(refund.getAuditTime());
        vo.setAuditorId(refund.getAuditorId());
        vo.setAuditRemark(refund.getAuditRemark());
        vo.setRefundTime(refund.getRefundTime());
        vo.setErrorMessage(refund.getErrorMessage());
        return vo;
    }
}
