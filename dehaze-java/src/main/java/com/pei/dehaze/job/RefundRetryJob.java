package com.pei.dehaze.job;

import com.pei.dehaze.security.util.SystemSecurityContext;
import com.pei.dehaze.service.OrderService;
import com.xxl.job.core.handler.annotation.XxlJob;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/**
 * 退款失败重试定时任务
 * 每30分钟扫描退款失败的记录，重新调用渠道退款接口，重试次数达上限则标记为最终失败。
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class RefundRetryJob {

    private final OrderService orderService;

    @XxlJob("retryFailedRefunds")
    public void retryFailedRefunds() {
        SystemSecurityContext.setSystemContext();
        try {
            log.info("开始执行退款失败重试...");
            orderService.retryFailedRefunds();
        } catch (Exception e) {
            log.error("退款失败重试处理失败", e);
        } finally {
            SystemSecurityContext.clearContext();
        }
    }
}
