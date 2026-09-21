package com.pei.dehaze.service;

import java.util.Map;

/** A2A 协议（Agent Card 发现端点） */
public interface AiA2aService {

    /** 按 Agent 主键动态生成 Agent Card（仅对外暴露且启用的普通 Agent） */
    Map<String, Object> getAgentCard(Long agentId, String baseUrl);
}
