package com.pei.dehaze.job;

import com.pei.dehaze.security.util.SystemSecurityContext;
import com.pei.dehaze.service.MemberService;
import com.xxl.job.core.handler.annotation.XxlJob;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/**
 * 会员过期降级定时任务
 * 每天凌晨2点扫描已过期且等级来源非 growth 的会员，按成长值重新计算等级。
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class MemberExpireJob {

    private final MemberService memberService;

    @XxlJob("processExpiredMembers")
    public void processExpiredMembers() {
        SystemSecurityContext.setSystemContext();
        try {
            log.info("开始执行会员过期降级检查...");
            memberService.processExpiredMembers();
        } catch (Exception e) {
            log.error("会员过期降级处理失败", e);
        } finally {
            SystemSecurityContext.clearContext();
        }
    }
}
