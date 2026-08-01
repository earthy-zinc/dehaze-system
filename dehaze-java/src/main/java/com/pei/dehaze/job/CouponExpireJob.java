package com.pei.dehaze.job;

import com.pei.dehaze.security.util.SystemSecurityContext;
import com.pei.dehaze.service.OrderService;
import com.xxl.job.core.handler.annotation.XxlJob;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/**
 * 用户优惠券过期处理定时任务
 * 扫描未使用且已过期的用户优惠券，标记为已过期。
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class CouponExpireJob {

    private final OrderService orderService;

    @XxlJob("expireUserCoupons")
    public void expireUserCoupons() {
        SystemSecurityContext.setSystemContext();
        try {
            log.debug("开始执行用户优惠券过期处理...");
            orderService.expireUserCoupons();
        } catch (Exception e) {
            log.error("用户优惠券过期处理失败", e);
        } finally {
            SystemSecurityContext.clearContext();
        }
    }
}
