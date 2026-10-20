package com.pei.clarity.model.read;

import lombok.Data;

/**
 * 会话消息状态行（异常标注数据源）
 *
 * @author clarity
 */
@Data
public class AnomalyStatusRead {

    private Long conversationId;

    private Integer status;
}
