package com.pei.dehaze.controller;

import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.service.AiA2aService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.support.ServletUriComponentsBuilder;

import java.util.Map;

/**
 * A2A 协议 Agent Card（挂载路径发现端点）。
 *
 * <p>JSON-RPC 入口（POST {agent}/a2a、POST /a2a）与全局发现（GET /.well-known/agent.json）
 * 属 B 类端点，统一由 {@link AiProxyController} 转发 python（行为事实源），本类只保留
 * 挂载路径下的 Agent Card 原生实现。
 */
@Tag(name = "AI对话-A2A协议")
@RestController
@RequiredArgsConstructor
public class AiA2aController {

    private final AiA2aService aiA2aService;

    @Operation(summary = "获取 Agent Card（动态生成）")
    @GetMapping("/api/v1/ai/agents/{agentId}/a2a/.well-known/agent.json")
    public Result<Map<String, Object>> getAgentCard(@PathVariable Long agentId) {
        return Result.success(aiA2aService.getAgentCard(agentId, baseUrl()));
    }

    private static String baseUrl() {
        return ServletUriComponentsBuilder.fromCurrentContextPath().build().toUriString();
    }
}
