package com.pei.clarity.model.read;

import lombok.Data;

/**
 * 会话计费消耗聚合行
 *
 * @author clarity
 */
@Data
public class ConversationConsumptionRead {

    private Long conversationId;

    private Long token;

    private Long credits;
}
