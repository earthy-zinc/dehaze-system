package com.pei.dehaze.model.vo;

import lombok.Data;

/**
 * 子 Agent 关联详情
 *
 * @author dehaze
 */
@Data
public class AiSubAgentVO {

    private Long agentId;

    private String agentName;

    private String agentCode;

    private String description;

    private Long endpointId;

    private Integer priority;
}
