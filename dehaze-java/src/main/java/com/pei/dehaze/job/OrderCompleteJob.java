package com.pei.dehaze.job;

import com.pei.dehaze.security.util.SystemSecurityContext;
import com.pei.dehaze.service.OrderService;
import com.xxl.job.core.handler.annotation.XxlJob;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/**
 * 订单到期自动完成定时任务
 * 扫描已支付且套餐到期的订单，归档为已完成。
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class OrderCompleteJob {

    private final OrderService orderService;

    @XxlJob("completeExpiredOrders")
    public void completeExpiredOrders() {
        SystemSecurityContext.setSystemContext();
        try {
            log.info("开始执行订单到期归档...");
            orderService.completeExpiredOrders();
        } catch (Exception e) {
            log.error("订单到期归档处理失败", e);
        } finally {
            SystemSecurityContext.clearContext();
        }
    }
}
