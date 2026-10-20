package com.pei.clarity.service;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.pei.clarity.mapper.SysMemberMapper;
import com.pei.clarity.model.entity.SysMember;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Component;

/**
 * 用户会员等级解析（level_0 → 0 … level_3 → 3，无会员记录按 0）。
 *
 * <p>等级用于模型 vip_level 过滤，是"哪些模型对该用户可见/可选"的唯一口径：
 * 启用模型列表（{@code GET /api/v1/ai/models/enabled}）与用户偏好默认模型校验共用本解析器。
 * 结果按用户缓存，与 python 同键（{@code user:level:{userId}}）。
 *
 * @author clarity
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class AiUserLevelResolver {

    private static final String USER_LEVEL_CACHE_PREFIX = "user:level:";
    private static final long USER_LEVEL_CACHE_TTL = 1800L;

    private static final Map<String, Integer> LEVEL_MAP =
            Map.of("level_0", 0, "level_1", 1, "level_2", 2, "level_3", 3);

    private final SysMemberMapper memberMapper;

    private final StringRedisTemplate redis;

    public int levelOf(Long userId) {
        if (userId == null) {
            return 0;
        }
        String key = USER_LEVEL_CACHE_PREFIX + userId;
        String cached = redis.opsForValue().get(key);
        if (cached != null) {
            try {
                return Integer.parseInt(cached);
            } catch (NumberFormatException e) {
                redis.delete(key);
            }
        }
        SysMember member = memberMapper.selectOne(new LambdaQueryWrapper<SysMember>()
                .eq(SysMember::getUserId, userId)
                .select(SysMember::getLevelCode));
        int level = member == null ? 0 : LEVEL_MAP.getOrDefault(member.getLevelCode(), 0);
        redis.opsForValue().set(key, String.valueOf(level), USER_LEVEL_CACHE_TTL, TimeUnit.SECONDS);
        return level;
    }
}
