package com.pei.dehaze.service.notify;

import com.pei.dehaze.model.entity.SysMessage;

/**
 * 消息推送渠道接口
 */
public interface MessagePushChannel {

    String getChannelType();

    boolean isAvailable();

    void send(SysMessage message, Long recipientId);

    final class SendResult {
        private final boolean success;
        private final String errorMessage;

        public SendResult(boolean success, String errorMessage) {
            this.success = success;
            this.errorMessage = errorMessage;
        }

        public boolean isSuccess() {
            return success;
        }

        public String getErrorMessage() {
            return errorMessage;
        }
    }
}
