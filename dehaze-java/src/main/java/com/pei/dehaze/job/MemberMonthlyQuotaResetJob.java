package com.pei.dehaze.job;

import com.pei.dehaze.security.util.SystemSecurityContext;
import com.pei.dehaze.service.MemberService;
import com.xxl.job.core.handler.annotation.XxlJob;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/**
 * 会员月度配额重置定时任务
 * 每月1日00:00执行，将上月配额使用情况归档到 sys_member_quota，并重置当月配额。
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class MemberMonthlyQuotaResetJob {

    private final MemberService memberService;

    @XxlJob("resetMonthlyQuota")
    public void resetMonthlyQuota() {
        SystemSecurityContext.setSystemContext();
        try {
            log.info("开始执行会员月度配额重置...");
            memberService.resetMonthlyQuota();
        } catch (Exception e) {
            log.error("会员月度配额重置失败", e);
        } finally {
            SystemSecurityContext.clearContext();
        }
    }
}
