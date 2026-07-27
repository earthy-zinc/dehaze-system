package com.pei.dehaze.service.notify;

import com.pei.dehaze.model.entity.SysMessage;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/**
 * 站内信渠道（消息已由 MessageService 写入 sys_message，此处仅记录日志）
 */
@Slf4j
@Component
public class InboxChannel implements MessagePushChannel {

    @Override
    public String getChannelType() {
        return "inbox";
    }

    @Override
    public boolean isAvailable() {
        return true;
    }

    @Override
    public void send(SysMessage message, Long recipientId) {
        log.debug("站内信已投递: messageId={}, recipientId={}", message.getId(), recipientId);
    }
}
