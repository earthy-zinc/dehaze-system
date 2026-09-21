package com.pei.dehaze.service.impl;

import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.AiA2aMapper;
import com.pei.dehaze.service.AiA2aService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** A2A Agent Card 实现（对齐 python a2a_server.build_agent_card） */
@Slf4j
@Service
@RequiredArgsConstructor
public class AiA2aServiceImpl implements AiA2aService {

    /** python 侧无已发布版本时的版本占位 */
    private static final String DEFAULT_VERSION = "0.0.0";

    private final AiA2aMapper a2aMapper;

    @Override
    @Transactional(readOnly = true)
    public Map<String, Object> getAgentCard(Long agentId, String baseUrl) {
        return buildCard(requireExposedAgent(agentId), baseUrl);
    }

    /** 取可对外服务的 Agent：不存在或未暴露/已禁用/子 Agent 均不可用（与 python 同口径） */
    private Map<String, Object> requireExposedAgent(Long agentId) {
        Map<String, Object> agent = a2aMapper.selectAgentForCard(agentId);
        if (agent == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "Agent 不存在");
        }
        if (!isOne(agent.get("status")) || !isOne(agent.get("isExposed")) || isOne(agent.get("isSubagent"))) {
            throw new BusinessException(ResultCode.OPERATION_NOT_ALLOW, "Agent 不可对外服务");
        }
        return agent;
    }

    private Map<String, Object> buildCard(Map<String, Object> agent, String baseUrl) {
        Long agentId = ((Number) agent.get("id")).longValue();
        Object name = agent.get("name");
        String agentCode = agent.get("agentCode") == null ? null : String.valueOf(agent.get("agentCode"));
        Object description = agent.get("description");
        Integer versionNo = a2aMapper.selectPublishedVersionNo(agentId);

        Map<String, Object> card = new LinkedHashMap<>();
        card.put("name", name == null || String.valueOf(name).isBlank() ? agentCode : name);
        card.put("description", description);
        card.put("version", versionNo == null ? DEFAULT_VERSION : String.valueOf(versionNo));
        card.put("url", baseUrl.replaceAll("/+$", "") + "/a2a");
        card.put("capabilities", Map.of("streaming", true, "pushNotifications", false));
        card.put("defaultInputModes", List.of("text", "file"));
        card.put("defaultOutputModes", List.of("text", "file"));
        card.put("skills", List.of(Map.of("name", agentCode, "description",
                description == null ? "" : description)));
        card.put("securitySchemes", Map.of("http", Map.of("scheme", "bearer")));
        card.put("security", List.of(Map.of("http", List.of())));
        return card;
    }

    private static boolean isOne(Object value) {
        return value instanceof Number number && number.intValue() == 1;
    }
}
