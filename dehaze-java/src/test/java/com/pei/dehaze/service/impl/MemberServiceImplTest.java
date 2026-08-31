package com.pei.dehaze.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.pei.dehaze.mapper.SysMemberGrowthLogMapper;
import com.pei.dehaze.mapper.SysMemberMapper;
import com.pei.dehaze.mapper.SysMemberQuotaMapper;
import com.pei.dehaze.mapper.SysMemberSignInMapper;
import com.pei.dehaze.mapper.SysUserMapper;
import com.pei.dehaze.model.entity.SysMember;
import com.pei.dehaze.model.entity.SysMemberSignIn;
import com.pei.dehaze.model.vo.SignInResultVO;
import com.pei.dehaze.security.util.SecurityUtils;
import com.pei.dehaze.service.MemberBenefitService;
import com.pei.dehaze.service.MessageService;
import com.pei.dehaze.service.SysDictService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.MockedStatic;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.data.redis.core.StringRedisTemplate;

import java.lang.reflect.Field;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.lenient;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.when;

/**
 * MemberServiceImpl 签到成长值字典化（sys_dict: member_growth_rules）单元测试。
 * <p>
 * 验证 sign_in() 从字典读取 sign_in_value / sign_in_streak_bonus 并正确应用（含连续 7 天奖励）。
 */
@DisplayName("MemberServiceImpl 签到成长值字典化单元测试")
@ExtendWith(MockitoExtension.class)
class MemberServiceImplTest {

    @Mock
    private SysUserMapper userMapper;
    @Mock
    private SysMemberGrowthLogMapper growthLogMapper;
    @Mock
    private SysMemberSignInMapper signInMapper;
    @Mock
    private SysMemberQuotaMapper quotaMapper;
    @Mock
    private MemberBenefitService memberBenefitService;
    @Mock
    private MessageService messageService;
    @Mock
    private StringRedisTemplate stringRedisTemplate;
    @Mock
    private SysDictService sysDictService;
    @Mock
    private SysMemberMapper memberMapper;

    private MemberServiceImpl service;

    @BeforeEach
    void setUp() throws Exception {
        service = new MemberServiceImpl(userMapper, growthLogMapper, signInMapper,
                quotaMapper, memberBenefitService, messageService, stringRedisTemplate, sysDictService);
        // 通过反射设置继承的 baseMapper，支撑 getMemberOrCreate 的 this.getOne
        Field baseMapperField = com.baomidou.mybatisplus.extension.service.impl.ServiceImpl.class
                .getDeclaredField("baseMapper");
        baseMapperField.setAccessible(true);
        baseMapperField.set(service, memberMapper);
    }

    private void stubExistingSignIn(int yesterdayContinuousDays) {
        when(signInMapper.selectCount(any(LambdaQueryWrapper.class))).thenReturn(0L);
        if (yesterdayContinuousDays > 0) {
            SysMemberSignIn yesterday = new SysMemberSignIn();
            yesterday.setContinuousDays(yesterdayContinuousDays);
            when(signInMapper.selectOne(any(LambdaQueryWrapper.class))).thenReturn(yesterday);
        } else {
            when(signInMapper.selectOne(any(LambdaQueryWrapper.class))).thenReturn(null);
        }
        when(signInMapper.insert(any(SysMemberSignIn.class))).thenReturn(1);
    }

    private void stubMember() {
        SysMember member = new SysMember();
        member.setUserId(1L);
        member.setLevelCode("level_0");
        member.setLevelSource("growth");
        member.setGrowthValue(0L);
        // getMemberOrCreate 的 this.getOne 走 baseMapper.selectOne(wrapper, true) 两参重载
        when(memberMapper.selectOne(any(LambdaQueryWrapper.class), eq(true))).thenReturn(member);
        when(memberBenefitService.listAllOrdered()).thenReturn(List.of());
    }

    @Test
    @DisplayName("signIn - 普通签到从字典读取 sign_in_value=3")
    void signIn_readsSignInValue() {
        stubExistingSignIn(0);
        stubMember();
        when(sysDictService.getIntValue(eq("member_growth_rules"), eq("sign_in_value"), any(Integer.class)))
                .thenReturn(3);
        lenient().when(sysDictService.getIntValue(eq("member_growth_rules"), eq("sign_in_streak_bonus"), any(Integer.class)))
                .thenReturn(20);

        try (MockedStatic<SecurityUtils> mocked = mockStatic(SecurityUtils.class)) {
            mocked.when(SecurityUtils::getUserId).thenReturn(1L);
            SignInResultVO vo = service.signIn();
            assertThat(vo.getGrowthValue()).isEqualTo(3);
            assertThat(vo.getBonusGrowth()).isZero();
        }
    }

    @Test
    @DisplayName("signIn - 连续第7天签到从字典读取 sign_in_value + sign_in_streak_bonus")
    void signIn_streakBonusReadsDict() {
        stubExistingSignIn(6); // 昨天连续6天，今天第7天触发奖励
        stubMember();
        when(sysDictService.getIntValue(eq("member_growth_rules"), eq("sign_in_value"), any(Integer.class)))
                .thenReturn(3);
        when(sysDictService.getIntValue(eq("member_growth_rules"), eq("sign_in_streak_bonus"), any(Integer.class)))
                .thenReturn(20);

        try (MockedStatic<SecurityUtils> mocked = mockStatic(SecurityUtils.class)) {
            mocked.when(SecurityUtils::getUserId).thenReturn(1L);
            SignInResultVO vo = service.signIn();
            assertThat(vo.getGrowthValue()).isEqualTo(3);
            assertThat(vo.getBonusGrowth()).isEqualTo(20);
        }
    }

    @Test
    @DisplayName("signIn - 字典缺键时回退默认值（sign_in_value=3、streak_bonus=20）")
    void signIn_missingDictKeyFallsBack() {
        stubExistingSignIn(6);
        stubMember();
        // 字典返回默认值，模拟缺键回退
        when(sysDictService.getIntValue(eq("member_growth_rules"), eq("sign_in_value"), any(Integer.class)))
                .thenReturn(3);
        when(sysDictService.getIntValue(eq("member_growth_rules"), eq("sign_in_streak_bonus"), any(Integer.class)))
                .thenReturn(20);

        try (MockedStatic<SecurityUtils> mocked = mockStatic(SecurityUtils.class)) {
            mocked.when(SecurityUtils::getUserId).thenReturn(1L);
            SignInResultVO vo = service.signIn();
            assertThat(vo.getGrowthValue()).isEqualTo(3);
            assertThat(vo.getBonusGrowth()).isEqualTo(20);
        }
    }
}
