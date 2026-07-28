package com.pei.dehaze.service.notify;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.pei.dehaze.mapper.SysNotificationSettingMapper;
import com.pei.dehaze.model.entity.SysMessage;
import com.pei.dehaze.model.entity.SysNotificationSetting;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Component;

import java.time.LocalTime;

@Slf4j
@Component
@RequiredArgsConstructor
public class PushChannel implements MessagePushChannel {

    private static final String DELAYED_PUSH_KEY = "msg:delayed_push:";
    private static final int CRITICAL_ALERT_PRIORITY = 4;

    private final SysNotificationSettingMapper settingMapper;
    private final StringRedisTemplate stringRedisTemplate;

    @Override
    public String getChannelType() {
        return "push";
    }

    @Override
    public boolean isAvailable() {
        return true;
    }

    @Override
    public void send(SysMessage message, Long recipientId) {
        if (!shouldPush(message, recipientId)) {
            return;
        }
        if (isInDndPeriod(recipientId) && !isCriticalAlert(message)) {
            enqueueDelayedPush(message, recipientId);
            log.info("消息进入免打扰延迟推送队列: messageId={}, recipientId={}", message.getId(), recipientId);
            return;
        }
        doPush(message, recipientId);
    }

    private boolean shouldPush(SysMessage message, Long recipientId) {
        SysNotificationSetting setting = getSetting(recipientId);
        if (setting == null || setting.getPushEnabled() == null || setting.getPushEnabled() == 0) {
            return false;
        }
        return true;
    }

    private boolean isInDndPeriod(Long recipientId) {
        SysNotificationSetting setting = getSetting(recipientId);
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

    private boolean isCriticalAlert(SysMessage message) {
        return message.getPriority() != null && message.getPriority() >= CRITICAL_ALERT_PRIORITY;
    }

    private void enqueueDelayedPush(SysMessage message, Long recipientId) {
        String key = DELAYED_PUSH_KEY + recipientId;
        String value = String.valueOf(message.getId());
        stringRedisTemplate.opsForList().rightPush(key, value);
    }

    private void doPush(SysMessage message, Long recipientId) {
        log.warn("APP推送通道暂未接入，跳过推送: messageId={}, recipientId={}", message.getId(), recipientId);
    }

    private SysNotificationSetting getSetting(Long recipientId) {
        return settingMapper.selectOne(new LambdaQueryWrapper<SysNotificationSetting>()
                .eq(SysNotificationSetting::getUserId, recipientId));
    }
}
