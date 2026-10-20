package com.pei.clarity.service;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.baomidou.mybatisplus.core.MybatisConfiguration;
import com.baomidou.mybatisplus.core.conditions.Wrapper;
import com.baomidou.mybatisplus.core.metadata.TableInfoHelper;
import com.pei.clarity.mapper.SysMemberMapper;
import com.pei.clarity.model.entity.SysMember;
import java.util.concurrent.TimeUnit;
import org.apache.ibatis.builder.MapperBuilderAssistant;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.data.redis.core.ValueOperations;

/**
 * 会员等级解析：模型 vip_level 过滤的唯一口径（启用模型列表与偏好校验共用）。
 */
@DisplayName("AiUserLevelResolver 会员等级解析")
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
class AiUserLevelResolverTest {

    private static final Long USER_ID = 21L;

    @Mock
    private SysMemberMapper memberMapper;

    @Mock
    private StringRedisTemplate redis;

    @Mock
    private ValueOperations<String, String> valueOperations;

    private AiUserLevelResolver resolver;

    @BeforeAll
    static void initMybatisPlusLambdaCache() {
        MybatisConfiguration configuration = new MybatisConfiguration();
        MapperBuilderAssistant assistant = new MapperBuilderAssistant(configuration, "user-level-test");
        assistant.setCurrentNamespace("com.pei.clarity.mapper.SysMemberMapper");
        TableInfoHelper.initTableInfo(assistant, SysMember.class);
    }

    @BeforeEach
    void setUp() {
        resolver = new AiUserLevelResolver(memberMapper, redis);
        when(redis.opsForValue()).thenReturn(valueOperations);
    }

    @Test
    @DisplayName("level_2 会员解析为 2，并按用户缓存等级")
    void resolvesLevelCodeAndCachesIt() {
        SysMember member = new SysMember();
        member.setUserId(USER_ID);
        member.setLevelCode("level_2");
        when(memberMapper.selectOne(any(Wrapper.class))).thenReturn(member);

        assertThat(resolver.levelOf(USER_ID)).isEqualTo(2);
        verify(valueOperations).set(eq("user:level:" + USER_ID), eq("2"), eq(1800L), eq(TimeUnit.SECONDS));
    }

    @Test
    @DisplayName("无会员记录按 level_0（0）处理")
    void absentMemberIsLevelZero() {
        when(memberMapper.selectOne(any(Wrapper.class))).thenReturn(null);

        assertThat(resolver.levelOf(USER_ID)).isZero();
        verify(valueOperations).set(anyString(), eq("0"), eq(1800L), eq(TimeUnit.SECONDS));
    }

    @Test
    @DisplayName("命中缓存直接返回，不再查库")
    void cachedLevelShortCircuits() {
        when(valueOperations.get("user:level:" + USER_ID)).thenReturn("3");

        assertThat(resolver.levelOf(USER_ID)).isEqualTo(3);
        verify(memberMapper, never()).selectOne(any(Wrapper.class));
    }

    @Test
    @DisplayName("缓存值非法时删除缓存并回源数据库")
    void invalidCacheFallsBackToDatabase() {
        when(valueOperations.get("user:level:" + USER_ID)).thenReturn("not-a-number");
        when(memberMapper.selectOne(any(Wrapper.class))).thenReturn(null);

        assertThat(resolver.levelOf(USER_ID)).isZero();
        verify(redis).delete("user:level:" + USER_ID);
    }

    @Test
    @DisplayName("无登录用户按 0 处理，不触达缓存与数据库")
    void nullUserIsLevelZero() {
        assertThat(resolver.levelOf(null)).isZero();
        verify(memberMapper, never()).selectOne(any(Wrapper.class));
    }
}
