package com.pei.dehaze.model.read;

import lombok.Data;

/**
 * 会话消息状态行（异常标注数据源）
 *
 * @author dehaze
 */
@Data
public class AnomalyStatusRead {

    private Long conversationId;

    private Integer status;
}
