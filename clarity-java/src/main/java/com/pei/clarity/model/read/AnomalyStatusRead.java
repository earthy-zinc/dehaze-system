package com.pei.clarity.model.read;

import lombok.Data;

/**
 * 会话消息状态行（异常标注数据源，取值为契约 {@code MessageStatus}）
 *
 * @author clarity
 */
@Data
public class AnomalyStatusRead {

    private Long conversationId;

    private String status;
}
