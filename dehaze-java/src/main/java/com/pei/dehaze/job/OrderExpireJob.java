package com.pei.dehaze.job;

import com.pei.dehaze.security.util.SystemSecurityContext;
import com.pei.dehaze.service.OrderService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
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

    @Scheduled(cron = "0 */5 * * * ?")
    public void expireOrders() {
        SystemSecurityContext.setSystemContext();
        try {
            log.info("开始执行订单超时取消检查...");
            orderService.expireOrders();
        } catch (Exception e) {
            log.error("订单超时取消处理失败", e);
        } finally {
            SystemSecurityContext.clearContext();
        }
    }
}
