package com.pei.dehaze.service.notify;

import com.pei.dehaze.model.entity.SysMessage;

/**
 * 消息推送渠道接口
 */
public interface MessagePushChannel {

    String getChannelType();

    boolean isAvailable();

    void send(SysMessage message, Long recipientId);
}
