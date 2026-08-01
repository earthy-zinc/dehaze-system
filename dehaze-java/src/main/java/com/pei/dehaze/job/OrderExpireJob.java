package com.pei.dehaze.job;

import com.pei.dehaze.security.util.SystemSecurityContext;
import com.pei.dehaze.service.OrderService;
import com.xxl.job.core.handler.annotation.XxlJob;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/**
 * 订单超时取消定时任务
 * 每5分钟扫描待支付订单，超过30分钟自动取消。
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class OrderExpireJob {

    private final OrderService orderService;

    @XxlJob("expireOrders")
    public void expireOrders() {
        SystemSecurityContext.setSystemContext();
        try {
            log.debug("开始执行订单超时取消检查...");
            orderService.expireOrders();
        } catch (Exception e) {
            log.error("订单超时取消处理失败", e);
        } finally {
            SystemSecurityContext.clearContext();
        }
    }
}
