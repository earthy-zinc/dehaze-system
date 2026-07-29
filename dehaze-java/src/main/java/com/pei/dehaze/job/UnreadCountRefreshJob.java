package com.pei.dehaze.job;

import com.pei.dehaze.security.util.SystemSecurityContext;
import com.pei.dehaze.service.MessageService;
import com.xxl.job.core.handler.annotation.XxlJob;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/**
 * 未读数缓存全量刷新定时任务
 * 每小时扫描所有活跃用户，重新计算未读数并刷新 Redis 缓存。
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class UnreadCountRefreshJob {

    private final MessageService messageService;

    @XxlJob("refreshUnreadCountCache")
    public void refreshUnreadCountCache() {
        SystemSecurityContext.setSystemContext();
        try {
            log.info("开始刷新未读数缓存...");
            messageService.refreshUnreadCountCache();
        } catch (Exception e) {
            log.error("未读数缓存刷新失败", e);
        } finally {
            SystemSecurityContext.clearContext();
        }
    }
}
