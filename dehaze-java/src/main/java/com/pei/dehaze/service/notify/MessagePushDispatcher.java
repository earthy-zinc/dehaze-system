package com.pei.dehaze.service.notify;

import com.pei.dehaze.model.entity.SysMessage;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * 推送分发器：根据渠道可用性分发消息推送
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class MessagePushDispatcher {

    private final List<MessagePushChannel> channels;

    @Async("pushTaskExecutor")
    public void dispatch(SysMessage message, Long recipientId) {
        for (MessagePushChannel channel : channels) {
            if (!channel.isAvailable()) {
                continue;
            }
            try {
                channel.send(message, recipientId);
            } catch (Exception e) {
                log.error("消息推送失败: channel={}, messageId={}, recipientId={}",
                        channel.getChannelType(), message.getId(), recipientId, e);
            }
        }
    }
}
