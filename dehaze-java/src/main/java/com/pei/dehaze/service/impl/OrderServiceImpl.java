package com.pei.dehaze.service.impl;

import cn.hutool.core.text.CharSequenceUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.annotation.AuditLog;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.SysAutoRenewMapper;
import com.pei.dehaze.mapper.SysCouponMapper;
import com.pei.dehaze.mapper.SysOrderMapper;
import com.pei.dehaze.mapper.SysPackageMapper;
import com.pei.dehaze.mapper.SysPaymentRecordMapper;
import com.pei.dehaze.mapper.SysRefundRecordMapper;
import com.pei.dehaze.mapper.SysUserCouponMapper;
import com.pei.dehaze.mapper.SysUserMapper;
import com.pei.dehaze.model.entity.SysAutoRenew;
import com.pei.dehaze.model.entity.SysCoupon;
import com.pei.dehaze.model.entity.SysMember;
import com.pei.dehaze.model.entity.SysMemberBenefit;
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
import com.pei.dehaze.service.MemberBenefitService;
import com.pei.dehaze.service.MemberService;
import com.pei.dehaze.service.OrderService;
import com.pei.dehaze.service.PackageService;
import com.pei.dehaze.service.payment.PaymentChannelService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.redisson.api.RLock;
import org.redisson.api.RedissonClient;
import org.springframework.data.redis.core.StringRedisTemplate;
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
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

@Slf4j
@Service
@RequiredArgsConstructor
public class OrderServiceImpl extends ServiceImpl<SysOrderMapper, SysOrder> implements OrderService {

    private static final int ORDER_TIMEOUT_MINUTES = 30;
    private static final int MAX_RENEW_FAIL_COUNT = 3;
    private static final DateTimeFormatter ORDER_NO_FORMAT = DateTimeFormatter.ofPattern("yyyyMMddHHmmss");
    private static final java.util.Set<String> PAY_METHODS = java.util.Set.of("wechat", "alipay", "balance", "combined");

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
    private final SysCouponMapper couponMapper;
    private final PackageService packageService;
    private final MemberService memberService;
    private final MemberBenefitService memberBenefitService;
    private final RedissonClient redissonClient;
    private final StringRedisTemplate stringRedisTemplate;
    private final List<PaymentChannelService> paymentChannelServices;
    private final com.fasterxml.jackson.databind.ObjectMapper objectMapper;

    @Override
    @Transactional(rollbackFor = Exception.class)
    @AuditLog(module = "order", action = "create", targetType = "order", targetIdSpel = "#result.orderNo", afterSpel = "#form")
    public PayResult create(OrderCreateForm form) {
        Long userId = SecurityUtils.getUserId();
        if (!PAY_METHODS.contains(form.getPayMethod())) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "不支持的支付方式");
        }
        SysPackage pkg = packageMapper.selectById(form.getPackageId());
        if (pkg == null) {
            throw new BusinessException(ResultCode.PACKAGE_NOT_FOUND);
        }
        if (pkg.getStatus() != 1) {
            throw new BusinessException(ResultCode.PACKAGE_OFF_SHELF);
        }
        String lockKey = "order:lock:" + userId + ":" + form.getPackageId();
        RLock lock = redissonClient.getLock(lockKey);
        boolean locked;
        try {
            locked = lock.tryLock(0, 5, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new BusinessException(ResultCode.REPEAT_SUBMIT_ERROR);
        }
        if (!locked) {
            throw new BusinessException(ResultCode.DUPLICATE_ORDER);
        }
        try {
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
        } finally {
            if (lock.isHeldByCurrentThread()) {
                lock.unlock();
            }
        }
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public PayResult pay(String orderNo, PayRequest request) {
        SysOrder order = getOrderByNo(orderNo);
        if (!order.getUserId().equals(SecurityUtils.getUserId())) {
            throw new BusinessException(ResultCode.ORDER_NOT_FOUND);
        }
        if (order.getStatus() != 1) {
            throw new BusinessException(ResultCode.ORDER_STATUS_INVALID);
        }
        if (order.getExpireTime() != null && order.getExpireTime().isBefore(LocalDateTime.now())) {
            throw new BusinessException(ResultCode.ORDER_EXPIRED);
        }
        order.setPayMethod(request.getPayMethod());
        this.updateById(order);
        invalidateOrderDetailCache(orderNo);
        PayResult result = new PayResult();
        result.setOrderNo(orderNo);
        result.setPayMethod(request.getPayMethod());

        if ("balance".equals(request.getPayMethod())) {
            completePayment(order, request.getPayMethod());
            result.setPaid(true);
        } else {
            PaymentChannelService channel = getPaymentChannel(request.getPayMethod());
            long amountFen = order.getPayableAmount() != null ? order.getPayableAmount() : 0L;
            PaymentChannelService.UnifiedOrderResult orderResult =
                    channel.unifiedOrder(orderNo, amountFen, order.getPackageName(), Map.of());
            if (!orderResult.success()) {
                throw new BusinessException(ResultCode.CALL_THIRD_PARTY_SERVICE_ERROR, orderResult.errorMessage());
            }
            createPaymentRecord(order, request.getPayMethod(), 1);
            result.setPayUrl(orderResult.payUrl());
            result.setQrCode(orderResult.qrCode());
            result.setPaid(false);
        }
        return result;
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public boolean handlePaymentCallback(String channelType, Map<String, String> params, String rawBody) {
        PaymentChannelService channel = getPaymentChannel(channelType);
        PaymentChannelService.CallbackVerifyResult verifyResult = channel.verifyCallback(params, rawBody);
        if (!verifyResult.success()) {
            log.warn("支付回调验签失败: channel={}, error={}", channelType, verifyResult.errorMessage());
            return false;
        }
        String orderNo = verifyResult.orderNo();
        if (orderNo == null) {
            log.warn("支付回调缺少订单号: channel={}", channelType);
            return false;
        }
        String lockKey = "payment:lock:" + orderNo;
        RLock lock = redissonClient.getLock(lockKey);
        boolean locked;
        try {
            locked = lock.tryLock(0, 10, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return false;
        }
        if (!locked) {
            log.info("支付回调重复，已跳过: orderNo={}", orderNo);
            return true;
        }
        try {
            SysOrder order = getOrderByNo(orderNo);
            if (order.getStatus() == 2 || order.getStatus() == 3) {
                return true;
            }
            if (order.getStatus() != 1) {
                log.warn("支付回调订单状态异常: orderNo={}, status={}", orderNo, order.getStatus());
                return false;
            }
            long expectedFen = order.getPayableAmount() != null ? order.getPayableAmount() : 0L;
            if (verifyResult.amountFen() > 0 && verifyResult.amountFen() != expectedFen) {
                log.error("支付回调金额不一致: orderNo={}, expected={}, actual={}", orderNo, expectedFen, verifyResult.amountFen());
                return false;
            }
            completePayment(order, channelType);
            return true;
        } finally {
            if (lock.isHeldByCurrentThread()) {
                lock.unlock();
            }
        }
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    @AuditLog(module = "order", action = "cancel", targetType = "order", targetIdSpel = "#orderNo", afterSpel = "{reason:#reason}")
    public void cancel(String orderNo, String reason) {
        SysOrder order = getOrderByNo(orderNo);
        if (!order.getUserId().equals(SecurityUtils.getUserId())) {
            throw new BusinessException(ResultCode.ORDER_NOT_FOUND);
        }
        if (order.getStatus() != 1) {
            throw new BusinessException(ResultCode.ORDER_STATUS_INVALID);
        }
        order.setStatus(4);
        order.setCancelReason(reason);
        this.updateById(order);
        if (order.getCouponId() != null) {
            unlockCoupon(order.getCouponId());
        }
        invalidateOrderDetailCache(orderNo);
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
        String cacheKey = "order:detail:" + orderNo;
        try {
            String cached = stringRedisTemplate.opsForValue().get(cacheKey);
            if (cached != null) {
                OrderDetailVO cachedVO = objectMapper.readValue(cached, OrderDetailVO.class);
                if (cachedVO.getUserId() == null || cachedVO.getUserId().equals(SecurityUtils.getUserId())) {
                    return cachedVO;
                }
            }
        } catch (Exception e) {
            log.warn("读取订单详情缓存失败: orderNo={}", orderNo, e);
        }

        SysOrder order = getOrderByNo(orderNo);
        if (!order.getUserId().equals(SecurityUtils.getUserId())) {
            throw new BusinessException(ResultCode.ORDER_NOT_FOUND);
        }
        OrderDetailVO vo = new OrderDetailVO();
        copyOrderPageFields(order, vo);
        vo.setUserId(order.getUserId());
        vo.setOriginalPrice(order.getOriginalPrice());
        vo.setDiscountAmount(order.getDiscountAmount());
        vo.setCouponAmount(order.getCouponAmount());
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

        try {
            stringRedisTemplate.opsForValue().set(cacheKey, objectMapper.writeValueAsString(vo), 10, TimeUnit.MINUTES);
        } catch (Exception e) {
            log.warn("写入订单详情缓存失败: orderNo={}", orderNo, e);
        }
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
        List<SysOrder> records = page.getRecords();
        Map<Long, SysUser> userMap;
        if (records.isEmpty()) {
            userMap = Collections.emptyMap();
        } else {
            userMap = userMapper.selectBatchIds(records.stream()
                            .map(SysOrder::getUserId).distinct().toList())
                    .stream().collect(Collectors.toMap(SysUser::getId, u -> u));
        }

        Page<OrderPageVO> result = new Page<>(page.getCurrent(), page.getSize(), page.getTotal());
        result.setRecords(records.stream().map(o -> toOrderPageVO(o, userMap.get(o.getUserId()))).toList());
        return result;
    }

    @Override
    public OrderStatsVO getStats(LocalDateTime startTime, LocalDateTime endTime) {
        OrderStatsVO stats = new OrderStatsVO();
        stats.setTotalOrders(baseMapper.countTotalOrders(startTime, endTime));
        stats.setTotalRevenue(baseMapper.sumRevenue(startTime, endTime));
        stats.setTotalRefund(refundRecordMapper.sumRefundAmount(startTime, endTime));
        stats.setRefundRate(stats.getTotalRevenue() > 0 ? (double) stats.getTotalRefund() / stats.getTotalRevenue() : 0.0);

        Map<String, Long> statusDist = new LinkedHashMap<>();
        for (String s : Arrays.asList("pending", "paid", "completed", "cancelled", "refunding", "refunded")) {
            statusDist.put(s, 0L);
        }
        for (Map<String, Object> row : baseMapper.selectStatusDistribution(startTime, endTime)) {
            Integer status = ((Number) row.get("status")).intValue();
            Long count = ((Number) row.get("cnt")).longValue();
            String name = ORDER_STATUS_MAP.get(status);
            if (name != null) {
                statusDist.put(name, count);
            }
        }
        stats.setStatusDistribution(statusDist);

        Map<String, Long> payMethodDist = new LinkedHashMap<>();
        for (String m : Arrays.asList("wechat", "alipay", "balance", "combined")) {
            payMethodDist.put(m, 0L);
        }
        for (Map<String, Object> row : baseMapper.selectPayMethodDistribution(startTime, endTime)) {
            String payMethod = (String) row.get("payMethod");
            Long count = ((Number) row.get("cnt")).longValue();
            if (payMethod != null && payMethodDist.containsKey(payMethod)) {
                payMethodDist.put(payMethod, count);
            }
        }
        stats.setPayMethodDistribution(payMethodDist);

        List<OrderStatsVO.PackageStatItem> pkgStats = new ArrayList<>();
        for (Map<String, Object> row : baseMapper.selectPackageDistribution(startTime, endTime)) {
            OrderStatsVO.PackageStatItem item = new OrderStatsVO.PackageStatItem();
            item.setPackageId(((Number) row.get("packageId")).longValue());
            item.setPackageName((String) row.get("packageName"));
            item.setCount(((Number) row.get("cnt")).longValue());
            item.setRevenue(((Number) row.get("revenue")).longValue());
            pkgStats.add(item);
        }
        stats.setPackageDistribution(pkgStats);

        List<OrderStatsVO.DailyStatItem> dailyStats = new ArrayList<>();
        for (Map<String, Object> row : baseMapper.selectDailyStats(startTime, endTime)) {
            OrderStatsVO.DailyStatItem item = new OrderStatsVO.DailyStatItem();
            item.setDate((String) row.get("date"));
            item.setCount(((Number) row.get("cnt")).longValue());
            item.setRevenue(((Number) row.get("revenue")).longValue());
            dailyStats.add(item);
        }
        stats.setDailyStats(dailyStats);
        return stats;
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    @AuditLog(module = "order", action = "refund_apply", targetType = "order", targetIdSpel = "#orderNo", afterSpel = "#form")
    public void applyRefund(String orderNo, RefundApplyForm form) {
        SysOrder order = getOrderByNo(orderNo);
        if (!order.getUserId().equals(SecurityUtils.getUserId())) {
            throw new BusinessException(ResultCode.ORDER_NOT_FOUND);
        }
        if (order.getStatus() != 2 && order.getStatus() != 3) {
            throw new BusinessException(ResultCode.ORDER_STATUS_INVALID);
        }
        SysRefundRecord existingRefund = refundRecordMapper.selectOne(new LambdaQueryWrapper<SysRefundRecord>()
                .eq(SysRefundRecord::getOrderId, order.getId())
                .orderByDesc(SysRefundRecord::getId)
                .last("LIMIT 1"));
        if (existingRefund != null) {
            throw new BusinessException(ResultCode.REFUND_ALREADY_EXISTS);
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
        invalidateOrderDetailCache(orderNo);
    }

    @Override
    public Page<RefundRecordVO> listRefunds(RefundPageQuery query) {
        Integer statusInt = refundStatusToInt(query.getStatus());
        long total = refundRecordMapper.countRefundPage(statusInt, query.getApplyTimeStart(),
                query.getApplyTimeEnd(), query.getOrderNo(), query.getKeywords());

        Page<RefundRecordVO> result = new Page<>(query.getPageNum(), query.getPageSize(), total);
        if (total == 0) {
            result.setRecords(Collections.emptyList());
            return result;
        }

        int offset = (int) Math.min((query.getPageNum() - 1) * query.getPageSize(), total);
        int limit = (int) Math.min(query.getPageSize(), total - offset);
        List<Map<String, Object>> rows = refundRecordMapper.selectRefundPageWithLimit(statusInt,
                query.getApplyTimeStart(), query.getApplyTimeEnd(), query.getOrderNo(), query.getKeywords(),
                offset, limit);

        List<RefundRecordVO> records = new ArrayList<>(rows.size());
        for (Map<String, Object> row : rows) {
            RefundRecordVO vo = new RefundRecordVO();
            vo.setId(((Number) row.get("id")).longValue());
            vo.setRefundNo((String) row.get("refund_no"));
            vo.setOrderId(((Number) row.get("order_id")).longValue());
            vo.setOrderNo((String) row.get("orderNo"));
            vo.setUserId(((Number) row.get("user_id")).longValue());
            vo.setUsername((String) row.get("username"));
            Number refundAmount = (Number) row.get("refund_amount");
            vo.setRefundAmount(refundAmount != null ? refundAmount.longValue() : 0L);
            vo.setReason((String) row.get("reason"));
            Number usedQuota = (Number) row.get("used_quota");
            vo.setUsedQuota(usedQuota != null ? usedQuota.intValue() : 0);
            Number statusVal = (Number) row.get("status");
            vo.setStatus(REFUND_STATUS_MAP.get(statusVal != null ? statusVal.intValue() : 0));
            vo.setChannel((String) row.get("channel"));
            vo.setChannelRefundNo((String) row.get("channel_refund_no"));
            vo.setApplyTime((java.time.LocalDateTime) row.get("apply_time"));
            vo.setAuditTime((java.time.LocalDateTime) row.get("audit_time"));
            Number auditorId = (Number) row.get("auditor_id");
            vo.setAuditorId(auditorId != null ? auditorId.longValue() : null);
            vo.setAuditRemark((String) row.get("audit_remark"));
            vo.setRefundTime((java.time.LocalDateTime) row.get("refund_time"));
            vo.setErrorMessage((String) row.get("error_message"));
            records.add(vo);
        }
        result.setRecords(records);
        return result;
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    @AuditLog(module = "order", action = "refund_approve", targetType = "order", targetIdSpel = "#refundId", afterSpel = "#form")
    public void approveRefund(Long refundId, RefundAuditForm form) {
        SysRefundRecord refund = refundRecordMapper.selectById(refundId);
        if (refund == null) {
            throw new BusinessException(ResultCode.REFUND_NOT_FOUND);
        }
        if (refund.getStatus() != 1) {
            throw new BusinessException(ResultCode.ORDER_STATUS_INVALID);
        }
        Long operatorId = SecurityUtils.getUserId();
        SysOrder order = this.getById(refund.getOrderId());
        if (order == null) {
            throw new BusinessException(ResultCode.ORDER_NOT_FOUND);
        }

        boolean refundOk = true;
        if (!"balance".equals(order.getPayMethod())) {
            try {
                PaymentChannelService channel = getPaymentChannel(order.getPayMethod());
                long totalFen = order.getPaidAmount() != null ? order.getPaidAmount() : 0L;
                long refundFen = refund.getRefundAmount() != null ? refund.getRefundAmount() : 0L;
                refundOk = channel.refund(order.getOrderNo(), refund.getRefundNo(),
                        totalFen, refundFen, refund.getReason());
            } catch (Exception e) {
                log.error("渠道退款失败: orderNo={}, refundNo={}", order.getOrderNo(), refund.getRefundNo(), e);
                refundOk = false;
            }
        }

        if (refundOk) {
            refund.setStatus(2);
            refund.setRefundTime(LocalDateTime.now());
            order.setStatus(6);
        } else {
            refund.setStatus(3);
            refund.setErrorMessage("渠道退款失败，待人工重试");
            order.setStatus(2);
        }
        refund.setAuditTime(LocalDateTime.now());
        refund.setAuditorId(operatorId);
        refund.setAuditRemark(form.getRemark());
        refundRecordMapper.updateById(refund);
        this.updateById(order);
        invalidateOrderDetailCache(order.getOrderNo());
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    @AuditLog(module = "order", action = "refund_reject", targetType = "order", targetIdSpel = "#refundId", afterSpel = "#form")
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
            invalidateOrderDetailCache(order.getOrderNo());
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
            if ("wechat".equals(order.getPayMethod()) || "alipay".equals(order.getPayMethod())) {
                try {
                    PaymentChannelService channel = getPaymentChannel(order.getPayMethod());
                    channel.closeOrder(order.getOrderNo());
                } catch (Exception e) {
                    log.warn("超时关单失败: orderNo={}", order.getOrderNo(), e);
                }
            }
            order.setStatus(4);
            order.setCancelReason("系统超时自动取消");
            this.updateById(order);
            if (order.getCouponId() != null) {
                unlockCoupon(order.getCouponId());
            }
            invalidateOrderDetailCache(order.getOrderNo());
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
                    renewal.setNextRenewTime(null);
                } else {
                    renewal.setNextRenewTime(LocalDateTime.now().plusHours(2));
                }
                autoRenewMapper.updateById(renewal);
            }
        }
        log.info("自动续费执行完成: 共处理{}条", renewals.size());
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void completeExpiredOrders() {
        List<SysOrder> expiredOrders = this.list(new LambdaQueryWrapper<SysOrder>()
                .eq(SysOrder::getStatus, 2)
                .lt(SysOrder::getPackageExpireTime, LocalDateTime.now()));
        for (SysOrder order : expiredOrders) {
            order.setStatus(3);
            this.updateById(order);
        }
        log.info("订单到期归档: 共处理{}条", expiredOrders.size());
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void expireUserCoupons() {
        List<SysUserCoupon> expiredCoupons = userCouponMapper.selectList(new LambdaQueryWrapper<SysUserCoupon>()
                .eq(SysUserCoupon::getStatus, 1)
                .lt(SysUserCoupon::getExpireTime, LocalDateTime.now()));
        for (SysUserCoupon coupon : expiredCoupons) {
            coupon.setStatus(3);
            userCouponMapper.updateById(coupon);
        }
        log.info("用户优惠券过期处理: 共处理{}条", expiredCoupons.size());
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void retryFailedRefunds() {
        int maxRetryCount = 3;
        List<SysRefundRecord> failedRefunds = refundRecordMapper.selectList(new LambdaQueryWrapper<SysRefundRecord>()
                .eq(SysRefundRecord::getStatus, 3)
                .lt(SysRefundRecord::getRetryCount, maxRetryCount));
        if (failedRefunds.isEmpty()) {
            log.info("退款失败重试: 无待处理记录");
            return;
        }
        int successCount = 0;
        int finalFailCount = 0;
        for (SysRefundRecord refund : failedRefunds) {
            SysOrder order = this.getById(refund.getOrderId());
            if (order == null) {
                log.warn("退款重试跳过: 退款记录{}对应订单不存在", refund.getId());
                continue;
            }
            int newRetryCount = (refund.getRetryCount() != null ? refund.getRetryCount() : 0) + 1;
            refund.setRetryCount(newRetryCount);

            boolean refundOk = true;
            String errorMessage = null;
            if (!"balance".equals(order.getPayMethod())) {
                try {
                    PaymentChannelService channel = getPaymentChannel(order.getPayMethod());
                    long totalFen = order.getPaidAmount() != null ? order.getPaidAmount() : 0L;
                    long refundFen = refund.getRefundAmount() != null ? refund.getRefundAmount() : 0L;
                    refundOk = channel.refund(order.getOrderNo(), refund.getRefundNo(),
                            totalFen, refundFen, refund.getReason());
                } catch (Exception e) {
                    log.error("渠道退款重试失败: orderNo={}, refundNo={}", order.getOrderNo(), refund.getRefundNo(), e);
                    refundOk = false;
                    errorMessage = e.getMessage();
                }
            }

            if (refundOk) {
                refund.setStatus(2);
                refund.setRefundTime(LocalDateTime.now());
                refund.setErrorMessage(null);
                order.setStatus(6);
                this.updateById(order);
                invalidateOrderDetailCache(order.getOrderNo());
                successCount++;
            } else {
                if (errorMessage == null) {
                    errorMessage = "渠道退款失败";
                }
                if (newRetryCount >= maxRetryCount) {
                    errorMessage = errorMessage + "（已达重试上限，转为最终失败）";
                    finalFailCount++;
                }
                refund.setErrorMessage(errorMessage);
            }
            refundRecordMapper.updateById(refund);
        }
        log.info("退款失败重试完成: 总数={}, 成功={}, 最终失败={}", failedRefunds.size(), successCount, finalFailCount);
    }

    private void executeSingleRenewal(SysAutoRenew renewal) {
        SysPackage pkg = packageMapper.selectById(renewal.getPackageId());
        if (pkg == null || pkg.getStatus() != 1) {
            throw new BusinessException(ResultCode.PACKAGE_NOT_FOUND);
        }
        long salePrice = pkg.getSalePrice() != null ? pkg.getSalePrice() : 0L;
        long payableAmount = (long) (salePrice * 0.95);

        if (!"balance".equals(renewal.getPayMethod())) {
            PaymentChannelService channel = getPaymentChannel(renewal.getPayMethod());
            boolean deductOk = channel.autoDeduct(renewal.getId() + "-" + System.currentTimeMillis(),
                    payableAmount, "自动续费-" + pkg.getName(), null);
            if (!deductOk) {
                throw new BusinessException(ResultCode.CALL_THIRD_PARTY_SERVICE_ERROR, "自动续费扣款失败");
            }
        }

        LocalDateTime now = LocalDateTime.now();
        LocalDateTime expireTime = activateMemberByPackage(renewal.getUserId(), pkg, now);

        SysOrder order = new SysOrder();
        order.setOrderNo(generateOrderNo());
        order.setUserId(renewal.getUserId());
        order.setPackageId(pkg.getId());
        order.setPackageName(pkg.getName());
        order.setPackageLevel(pkg.getLevelCode());
        order.setPeriodDays(pkg.getPeriodDays());
        order.setOriginalPrice(pkg.getOriginalPrice());
        order.setDiscountAmount(salePrice - payableAmount);
        order.setCouponAmount(0L);
        order.setPayableAmount(payableAmount);
        order.setPaidAmount(payableAmount);
        order.setPayMethod(renewal.getPayMethod());
        order.setStatus(2);
        order.setPaidTime(now);
        order.setEffectiveTime(now);
        order.setPackageExpireTime(expireTime);
        order.setIsAutoRenew(1);
        this.save(order);

        createPaymentRecord(order, renewal.getPayMethod(), 2);
        updatePackageSalesCount(pkg.getId());

        renewal.setLastRenewOrderId(order.getId());
        renewal.setNextRenewTime(expireTime);
        renewal.setFailCount(0);
        autoRenewMapper.updateById(renewal);
    }

    private PaymentChannelService getPaymentChannel(String channelType) {
        return paymentChannelServices.stream()
                .filter(ch -> ch.getChannelType().equals(channelType))
                .findFirst()
                .orElseThrow(() -> new BusinessException(ResultCode.PARAM_ERROR, "不支持的支付渠道: " + channelType));
    }

    private void completePayment(SysOrder order, String payMethod) {
        LocalDateTime now = LocalDateTime.now();
        SysPackage pkg = packageMapper.selectById(order.getPackageId());
        LocalDateTime expireTime;
        if (pkg != null) {
            expireTime = activateMemberByPackage(order.getUserId(), pkg, now);
        } else {
            expireTime = now.plusDays(order.getPeriodDays() != null ? order.getPeriodDays() : 30);
        }
        order.setStatus(2);
        order.setPaidTime(now);
        order.setEffectiveTime(now);
        order.setPaidAmount(order.getPayableAmount());
        order.setPackageExpireTime(expireTime);
        this.updateById(order);

        createPaymentRecord(order, payMethod, 2);
        updatePackageSalesCount(order.getPackageId());
        if (order.getCouponId() != null) {
            consumeCoupon(order.getCouponId(), order.getId());
        }
        invalidateOrderDetailCache(order.getOrderNo());
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

    private LocalDateTime activateMemberByPackage(Long userId, SysPackage pkg, LocalDateTime now) {
        SysMember member = memberService.getOne(new LambdaQueryWrapper<SysMember>()
                .eq(SysMember::getUserId, userId));
        LocalDateTime baseTime = now;
        if (member != null && member.getExpireTime() != null && member.getExpireTime().isAfter(now)) {
            baseTime = member.getExpireTime();
        }
        LocalDateTime expireTime = baseTime.plusDays(pkg.getPeriodDays() != null ? pkg.getPeriodDays() : 30);
        SysMemberBenefit benefit = memberBenefitService.getByLevelCode(pkg.getLevelCode());

        if (member == null) {
            member = new SysMember();
            member.setUserId(userId);
            member.setLevelCode(pkg.getLevelCode());
            member.setLevelSource("package");
            member.setGrowthValue(0L);
            member.setTotalConsumption(pkg.getSalePrice() != null ? pkg.getSalePrice() : 0L);
            member.setExpireTime(expireTime);
            member.setBecomeMemberTime(now);
            member.setStatus(1);
            if (benefit != null) {
                member.setMonthlyDehazeQuota(benefit.getMonthlyDehazeQuota());
                member.setMonthlyEvaluateQuota(benefit.getMonthlyEvaluateQuota());
            }
            member.setMonthlyDehazeUsed(0);
            member.setMonthlyEvaluateUsed(0);
            member.setQuotaResetMonth(Integer.parseInt(now.format(DateTimeFormatter.ofPattern("yyyyMM"))));
            memberService.save(member);
        } else {
            LambdaUpdateWrapper<SysMember> wrapper = new LambdaUpdateWrapper<SysMember>()
                    .eq(SysMember::getUserId, userId)
                    .set(SysMember::getLevelCode, pkg.getLevelCode())
                    .set(SysMember::getLevelSource, "package")
                    .set(SysMember::getExpireTime, expireTime)
                    .set(SysMember::getStatus, 1)
                    .set(SysMember::getTotalConsumption,
                            (member.getTotalConsumption() != null ? member.getTotalConsumption() : 0L)
                                    + (pkg.getSalePrice() != null ? pkg.getSalePrice() : 0L));
            if (benefit != null) {
                wrapper.set(SysMember::getMonthlyDehazeQuota, benefit.getMonthlyDehazeQuota());
                wrapper.set(SysMember::getMonthlyEvaluateQuota, benefit.getMonthlyEvaluateQuota());
            }
            memberService.update(wrapper);
        }
        invalidateMemberCache(userId);
        return expireTime;
    }

    private void lockCoupon(Long userCouponId, Long userId) {
        SysUserCoupon userCoupon = userCouponMapper.selectById(userCouponId);
        if (userCoupon == null || !userCoupon.getUserId().equals(userId)) {
            throw new BusinessException(ResultCode.COUPON_NOT_FOUND);
        }
        LambdaUpdateWrapper<SysUserCoupon> wrapper = new LambdaUpdateWrapper<SysUserCoupon>()
                .eq(SysUserCoupon::getId, userCouponId)
                .eq(SysUserCoupon::getStatus, 1)
                .set(SysUserCoupon::getStatus, 4);
        int rows = userCouponMapper.update(null, wrapper);
        if (rows == 0) {
            throw new BusinessException(ResultCode.COUPON_ALREADY_USED);
        }
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
            couponMapper.update(null, new LambdaUpdateWrapper<SysCoupon>()
                    .eq(SysCoupon::getId, userCoupon.getCouponId())
                    .setSql("used_qty = used_qty + 1"));
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

    private void invalidateOrderDetailCache(String orderNo) {
        stringRedisTemplate.delete("order:detail:" + orderNo);
    }

    private void invalidateMemberCache(Long userId) {
        stringRedisTemplate.delete("member:level:" + userId);
        stringRedisTemplate.delete("member:quota:" + userId + ":dehaze");
        stringRedisTemplate.delete("member:quota:" + userId + ":evaluate");
    }

    private String generateOrderNo() {
        return "DH" + LocalDateTime.now().format(ORDER_NO_FORMAT) + ThreadLocalRandom.current().nextInt(100000, 999999);
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
