package com.pei.dehaze.job;

import com.pei.dehaze.security.util.SystemSecurityContext;
import com.pei.dehaze.service.OrderService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

/**
 * 自动续费定时任务
 * 每日凌晨3点扫描到期自动续费记录，自动创建新订单并完成支付。
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class AutoRenewJob {

    private final OrderService orderService;

    @Scheduled(cron = "0 0 3 * * ?")
    public void executeRenewal() {
        SystemSecurityContext.setSystemContext();
        try {
            log.info("开始执行自动续费...");
            orderService.executeRenewal();
        } catch (Exception e) {
            log.error("自动续费处理失败", e);
        } finally {
            SystemSecurityContext.clearContext();
        }
    }
}
