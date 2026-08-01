package com.pei.dehaze.job;

import com.pei.dehaze.security.util.SystemSecurityContext;
import com.pei.dehaze.service.MemberService;
import com.xxl.job.core.handler.annotation.XxlJob;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/**
 * 会员到期预警定时任务
 * 每日09:00扫描 expire_time 在未来 7/3/1 天的会员，推送续费提醒站内信。
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class MemberExpireReminderJob {

    private final MemberService memberService;

    @XxlJob("sendExpireReminders")
    public void sendExpireReminders() {
        SystemSecurityContext.setSystemContext();
        try {
            log.debug("开始执行会员到期预警...");
            memberService.sendExpireReminders();
        } catch (Exception e) {
            log.error("会员到期预警处理失败", e);
        } finally {
            SystemSecurityContext.clearContext();
        }
    }
}
