package com.pei.dehaze.model.read;

import lombok.Data;

/**
 * Agent 关联计数行
 *
 * @author dehaze
 */
@Data
public class AgentRefCountRead {

    private Long agentId;

    private Long cnt;
}
