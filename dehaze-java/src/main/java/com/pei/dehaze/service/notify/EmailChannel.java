package com.pei.dehaze.service.notify;

import com.pei.dehaze.model.entity.SysMessage;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/**
 * 邮件渠道（依赖第三方邮件服务，暂不可用）
 */
@Slf4j
@Component
public class EmailChannel implements MessagePushChannel {

    @Override
    public String getChannelType() {
        return "email";
    }

    @Override
    public boolean isAvailable() {
        return false;
    }

    @Override
    public void send(SysMessage message, Long recipientId) {
        log.warn("邮件通道暂未接入，跳过发送: messageId={}, recipientId={}", message.getId(), recipientId);
    }
}
