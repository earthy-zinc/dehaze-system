package com.pei.clarity.job;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.baomidou.mybatisplus.core.MybatisConfiguration;
import com.baomidou.mybatisplus.core.metadata.TableInfoHelper;
import com.baomidou.mybatisplus.core.toolkit.GlobalConfigUtils;
import com.pei.clarity.common.exception.BusinessException;
import com.pei.clarity.common.result.ResultCode;
import com.pei.clarity.mapper.SysMemberMapper;
import com.pei.clarity.model.entity.SysMember;
import com.pei.clarity.model.entity.SysMemberBenefit;
import com.pei.clarity.service.AiBillingService;
import com.pei.clarity.service.MemberBenefitService;
import java.util.List;
import org.apache.ibatis.builder.MapperBuilderAssistant;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

/**
 * VIP 积分赠送任务（grantVipMonthlyGift / clearVipGiftExpire）调度编排口径。
 *
 * <p>任务名与 python、Go 端一致（config/sql/schema-external/xxl_job.sql 已注册），此处锁定两条编排纪律：
 * 仅对启用且 vip_gift_credits&gt;0 的等级发放；单用户清零失败不得中断整批。
 */
@DisplayName("MemberVipGiftJob VIP 积分赠送任务")
@ExtendWith(MockitoExtension.class)
class MemberVipGiftJobTest {

    @Mock
    private MemberBenefitService memberBenefitService;

    @Mock
    private SysMemberMapper memberMapper;

    @Mock
    private AiBillingService aiBillingService;

    @InjectMocks
    private MemberVipGiftJob job;

    @BeforeAll
    static void initTableInfo() {
        MybatisConfiguration configuration = new MybatisConfiguration();
        GlobalConfigUtils.setGlobalConfig(configuration, GlobalConfigUtils.defaults());
        TableInfoHelper.initTableInfo(new MapperBuilderAssistant(configuration, ""), SysMember.class);
    }

    @Test
    @DisplayName("月度赠送：仅向启用且 vip_gift_credits>0 的等级发放对应额度")
    void grantVipMonthlyGift_onlyGiftLevels() {
        SysMemberBenefit giftLevel = benefit("level_2", 1, 300L);
        SysMemberBenefit disabledLevel = benefit("level_3", 0, 500L);
        SysMemberBenefit zeroLevel = benefit("level_1", 1, 0L);
        when(memberBenefitService.listAllOrdered()).thenReturn(List.of(giftLevel, disabledLevel, zeroLevel));
        SysMember member = new SysMember();
        member.setId(11L);
        member.setUserId(7L);
        member.setLevelCode("level_2");
        member.setStatus(1);
        when(memberMapper.selectList(any())).thenReturn(List.of(member)).thenReturn(List.of());

        job.grantVipMonthlyGift();

        verify(aiBillingService).grantVipMonthlyGift(7L, 300);
    }

    @Test
    @DisplayName("月末清零：逐用户清零，单用户失败不阻断整批")
    void clearVipGiftExpire_continuesAfterUserFailure() {
        when(aiBillingService.listVipGiftUserIds()).thenReturn(List.of(7L, 8L, 9L));
        when(aiBillingService.expireVipGiftCredits(7L)).thenReturn(200L);
        when(aiBillingService.expireVipGiftCredits(8L))
                .thenThrow(new BusinessException(ResultCode.BUSINESS_ERROR, "余额落库失败"));
        when(aiBillingService.expireVipGiftCredits(9L)).thenReturn(0L);

        job.clearVipGiftExpire();

        verify(aiBillingService).expireVipGiftCredits(7L);
        verify(aiBillingService).expireVipGiftCredits(8L);
        verify(aiBillingService).expireVipGiftCredits(9L);
    }

    @Test
    @DisplayName("月末清零：当月无赠送用户时不调用清零")
    void clearVipGiftExpire_skipsWithoutGiftUsers() {
        when(aiBillingService.listVipGiftUserIds()).thenReturn(List.of());

        job.clearVipGiftExpire();

        verify(aiBillingService, never()).expireVipGiftCredits(anyLong());
    }

    private static SysMemberBenefit benefit(String levelCode, int status, Long vipGiftCredits) {
        SysMemberBenefit benefit = new SysMemberBenefit();
        benefit.setLevelCode(levelCode);
        benefit.setStatus(status);
        benefit.setVipGiftCredits(vipGiftCredits);
        return benefit;
    }
}
