package com.pei.dehaze.model.read;

import lombok.Data;

/**
 * 会话计费消耗聚合行
 *
 * @author dehaze
 */
@Data
public class ConversationConsumptionRead {

    private Long conversationId;

    private Long token;

    private Long credits;
}
