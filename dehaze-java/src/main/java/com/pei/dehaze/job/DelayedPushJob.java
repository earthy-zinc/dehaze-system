package com.pei.dehaze.job;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.pei.dehaze.mapper.SysNotificationSettingMapper;
import com.pei.dehaze.model.entity.SysMessage;
import com.pei.dehaze.model.entity.SysNotificationSetting;
import com.pei.dehaze.security.util.SystemSecurityContext;
import com.pei.dehaze.service.MessageService;
import com.pei.dehaze.service.notify.MessagePushDispatcher;
import com.xxl.job.core.handler.annotation.XxlJob;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Component;

import java.time.LocalTime;
import java.util.Set;

@Slf4j
@Component
@RequiredArgsConstructor
public class DelayedPushJob {

    private static final String DELAYED_PUSH_KEY_PREFIX = "msg:delayed_push:";

    private final StringRedisTemplate stringRedisTemplate;
    private final SysNotificationSettingMapper settingMapper;
    private final MessageService messageService;
    private final MessagePushDispatcher pushDispatcher;

    @XxlJob("processDelayedPush")
    public void processDelayedPush() {
        SystemSecurityContext.setSystemContext();
        try {
            Set<String> keys = stringRedisTemplate.keys(DELAYED_PUSH_KEY_PREFIX + "*");
            if (keys == null || keys.isEmpty()) {
                return;
            }
            for (String key : keys) {
                processUserDelayedPush(key);
            }
        } finally {
            SystemSecurityContext.clearContext();
        }
    }

    private void processUserDelayedPush(String key) {
        Long userId = extractUserId(key);
        if (userId == null) {
            return;
        }
        if (isStillInDnd(userId)) {
            return;
        }
        String messageId;
        while ((messageId = stringRedisTemplate.opsForList().leftPop(key)) != null) {
            try {
                SysMessage message = messageService.getById(Long.parseLong(messageId));
                if (message != null) {
                    pushDispatcher.dispatch(message, userId);
                    log.debug("处理免打扰延迟推送: messageId={}, recipientId={}", message.getId(), userId);
                }
            } catch (Exception e) {
                log.error("延迟推送处理失败: messageId={}", messageId, e);
            }
        }
    }

    private Long extractUserId(String key) {
        try {
            return Long.parseLong(key.substring(DELAYED_PUSH_KEY_PREFIX.length()));
        } catch (NumberFormatException e) {
            return null;
        }
    }

    private boolean isStillInDnd(Long userId) {
        SysNotificationSetting setting = settingMapper.selectOne(new LambdaQueryWrapper<SysNotificationSetting>()
                .eq(SysNotificationSetting::getUserId, userId));
        if (setting == null || setting.getDndEnabled() == null || setting.getDndEnabled() == 0) {
            return false;
        }
        LocalTime now = LocalTime.now();
        LocalTime start = setting.getDndStart();
        LocalTime end = setting.getDndEnd();
        if (start == null || end == null) {
            return false;
        }
        if (start.isBefore(end)) {
            return !now.isBefore(start) && !now.isAfter(end);
        }
        return !now.isBefore(start) || !now.isAfter(end);
    }
}
