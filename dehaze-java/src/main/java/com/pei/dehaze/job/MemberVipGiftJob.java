package com.pei.dehaze.job;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.pei.dehaze.mapper.SysMemberMapper;
import com.pei.dehaze.model.entity.SysMember;
import com.pei.dehaze.model.entity.SysMemberBenefit;
import com.pei.dehaze.security.util.SystemSecurityContext;
import com.pei.dehaze.service.AiBillingService;
import com.pei.dehaze.service.MemberBenefitService;
import com.xxl.job.core.handler.annotation.XxlJob;
import java.util.List;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/**
 * VIP 积分赠送任务：按月发放（vip_gift）+ 月末清零未用部分（vip_gift_expire）。
 *
 * <p>承接原会员月度配额权益（设计 §9.1）：会员权益改为月度积分赠送，复用 vip_gift 积分来源。
 * 任务名 grantVipMonthlyGift / clearVipGiftExpire 与 python、Go 端保持一致（调度行见 config/sql/schema-external/xxl_job.sql）。
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class MemberVipGiftJob {

    private static final int BATCH_SIZE = 500;

    private final MemberBenefitService memberBenefitService;
    private final SysMemberMapper memberMapper;
    private final AiBillingService aiBillingService;

    @XxlJob("grantVipMonthlyGift")
    public void grantVipMonthlyGift() {
        SystemSecurityContext.setSystemContext();
        try {
            List<SysMemberBenefit> giftLevels = memberBenefitService.listAllOrdered().stream()
                    .filter(b -> Integer.valueOf(1).equals(b.getStatus())
                            && b.getVipGiftCredits() != null
                            && b.getVipGiftCredits() > 0)
                    .toList();

            int granted = 0;
            for (SysMemberBenefit benefit : giftLevels) {
                int amount = benefit.getVipGiftCredits().intValue();
                long lastId = 0;
                while (true) {
                    List<SysMember> members = memberMapper.selectList(new LambdaQueryWrapper<SysMember>()
                            .eq(SysMember::getLevelCode, benefit.getLevelCode())
                            .eq(SysMember::getStatus, 1)
                            .gt(SysMember::getId, lastId)
                            .orderByAsc(SysMember::getId)
                            .last("LIMIT " + BATCH_SIZE));
                    if (members.isEmpty()) {
                        break;
                    }
                    for (SysMember member : members) {
                        try {
                            aiBillingService.grantVipMonthlyGift(member.getUserId(), amount);
                            granted++;
                        } catch (Exception e) {
                            log.warn(
                                    "VIP月度赠送失败 userId={} level={}: {}",
                                    member.getUserId(),
                                    benefit.getLevelCode(),
                                    e.getMessage());
                        }
                    }
                    lastId = members.get(members.size() - 1).getId();
                }
            }
            log.debug("VIP月度赠送完成: 等级数={}, 发放用户数={}", giftLevels.size(), granted);
        } catch (Exception e) {
            log.error("VIP月度赠送任务失败", e);
        } finally {
            SystemSecurityContext.clearContext();
        }
    }

    @XxlJob("clearVipGiftExpire")
    public void clearVipGiftExpire() {
        SystemSecurityContext.setSystemContext();
        try {
            List<Long> giftUserIds = aiBillingService.listVipGiftUserIds();
            int cleared = 0;
            for (Long userId : giftUserIds) {
                try {
                    if (aiBillingService.expireVipGiftCredits(userId) > 0) {
                        cleared++;
                    }
                } catch (Exception e) {
                    log.warn("VIP赠送积分清零失败 userId={}: {}", userId, e.getMessage());
                }
            }
            log.debug("VIP赠送积分月末清零完成: 涉及用户={}, 清零={}", giftUserIds.size(), cleared);
        } catch (Exception e) {
            log.error("VIP赠送积分月末清零任务失败", e);
        } finally {
            SystemSecurityContext.clearContext();
        }
    }
}
