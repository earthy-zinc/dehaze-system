package com.pei.dehaze.service.notify;

import com.pei.dehaze.model.entity.SysMessage;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/**
 * APP推送渠道（依赖第三方推送平台，暂不可用）
 */
@Slf4j
@Component
public class PushChannel implements MessagePushChannel {

    @Override
    public String getChannelType() {
        return "push";
    }

    @Override
    public boolean isAvailable() {
        return false;
    }

    @Override
    public void send(SysMessage message, Long recipientId) {
        log.warn("APP推送通道暂未接入，跳过推送: messageId={}, recipientId={}", message.getId(), recipientId);
    }
}
