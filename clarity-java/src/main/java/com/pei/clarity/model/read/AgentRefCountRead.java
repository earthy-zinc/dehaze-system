package com.pei.clarity.model.read;

import lombok.Data;

/**
 * Agent 关联计数行
 *
 * @author clarity
 */
@Data
public class AgentRefCountRead {

    private Long agentId;

    private Long cnt;
}
