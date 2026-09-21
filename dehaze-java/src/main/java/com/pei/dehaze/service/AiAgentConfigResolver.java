package com.pei.dehaze.service;

import com.pei.dehaze.mapper.AiInsightMapper;
import com.pei.dehaze.model.read.DictItemRead;
import com.pei.dehaze.model.vo.AiAgentConfigDefaultsVO;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Agent 配置三级合并解析器（系统默认 ← Agent 配置 ← 会话级覆盖）。
 *
 * <p>对齐 dehaze-python {@code agent_config_resolver}：推理参数系统默认值为代码常量（无管理员
 * 运行时修改场景），护栏默认值取 sys_dict {@code ai_guardrail_defaults}（平铺 name 转嵌套结构，
 * 安全开关有运维调整场景）。高优先级层覆盖低优先级同名项。
 *
 * @author dehaze
 */
@Component
@RequiredArgsConstructor
public class AiAgentConfigResolver {

    private static final String GUARDRAIL_DEFAULTS_DICT = "ai_guardrail_defaults";

    /** 推理参数系统默认值（代码常量，勿与 sys_dict 混用） */
    private static final Map<String, Object> REASONING_DEFAULTS = Map.of(
            "max_steps_react", 20,
            "max_steps_plan", 30,
            "max_steps_reflexion", 15,
            "max_iterations_reflexion", 3,
            "reflexion_threshold", 0.8,
            "max_parallel", 5,
            "tool_timeout", 60,
            "token_budget", 500000,
            "retry_max", 2);

    private final AiInsightMapper insightMapper;

    /**
     * 推理参数系统默认值对外契约（Agent 配置表单展示「空值继承」用）
     */
    public AiAgentConfigDefaultsVO reasoningDefaults() {
        AiAgentConfigDefaultsVO defaults = new AiAgentConfigDefaultsVO();
        defaults.setMaxStepsReact((Integer) REASONING_DEFAULTS.get("max_steps_react"));
        defaults.setMaxStepsPlan((Integer) REASONING_DEFAULTS.get("max_steps_plan"));
        defaults.setMaxStepsReflexion((Integer) REASONING_DEFAULTS.get("max_steps_reflexion"));
        defaults.setMaxIterationsReflexion((Integer) REASONING_DEFAULTS.get("max_iterations_reflexion"));
        defaults.setReflexionThreshold((Double) REASONING_DEFAULTS.get("reflexion_threshold"));
        defaults.setMaxParallel((Integer) REASONING_DEFAULTS.get("max_parallel"));
        defaults.setToolTimeout((Integer) REASONING_DEFAULTS.get("tool_timeout"));
        defaults.setTokenBudget((Integer) REASONING_DEFAULTS.get("token_budget"));
        defaults.setRetryMax((Integer) REASONING_DEFAULTS.get("retry_max"));
        return defaults;
    }

    /**
     * 合并生效配置：推理参数平铺于顶层，护栏汇总于 guardrails 子对象
     */
    public Map<String, Object> resolve(Map<String, Object> agentConfig) {
        Map<String, Object> agentCfg = agentConfig == null ? Map.of() : agentConfig;
        Map<String, Object> reasoning = new LinkedHashMap<>(REASONING_DEFAULTS);
        agentCfg.forEach((key, value) -> {
            if (!"guardrails".equals(key)) {
                reasoning.put(key, value);
            }
        });
        Map<String, Object> guardrails = mergeGuardrails(loadGuardrailDefaults(), agentCfg.get("guardrails"));
        reasoning.put("guardrails", guardrails);
        return reasoning;
    }

    /**
     * 护栏系统默认值：sys_dict 平铺 name（如 prompt_injection.enabled）组装为嵌套 {规则名: {参数}}
     */
    private Map<String, Object> loadGuardrailDefaults() {
        Map<String, Object> nested = new LinkedHashMap<>();
        List<DictItemRead> items = insightMapper.listEnabledDictItems(GUARDRAIL_DEFAULTS_DICT);
        for (DictItemRead item : items) {
            String[] parts = item.getName().split("\\.");
            Map<String, Object> node = nested;
            for (int i = 0; i < parts.length - 1; i++) {
                Object child = node.get(parts[i]);
                if (!(child instanceof Map)) {
                    child = new LinkedHashMap<String, Object>();
                    node.put(parts[i], child);
                }
                @SuppressWarnings("unchecked")
                Map<String, Object> next = (Map<String, Object>) child;
                node = next;
            }
            node.put(parts[parts.length - 1], coerceScalar(item.getValue()));
        }
        return nested;
    }

    private Map<String, Object> mergeGuardrails(Map<String, Object> defaults, Object override) {
        Map<String, Object> result = new LinkedHashMap<>();
        defaults.forEach((key, value) -> result.put(key, value instanceof Map
                ? new LinkedHashMap<>((Map<String, Object>) value) : value));
        if (!(override instanceof Map)) {
            return result;
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> layer = (Map<String, Object>) override;
        layer.forEach((ruleName, rule) -> {
            if (rule instanceof Map && result.get(ruleName) instanceof Map) {
                Map<String, Object> merged = new LinkedHashMap<>((Map<String, Object>) result.get(ruleName));
                merged.putAll((Map<String, Object>) rule);
                result.put(ruleName, merged);
            } else {
                result.put(ruleName, rule);
            }
        });
        return result;
    }

    private Object coerceScalar(String raw) {
        if (raw == null) {
            return null;
        }
        String text = raw.trim();
        if ("true".equalsIgnoreCase(text) || "false".equalsIgnoreCase(text)) {
            return Boolean.parseBoolean(text);
        }
        try {
            return Integer.parseInt(text);
        } catch (NumberFormatException ignored) {
            // 继续按浮点解析
        }
        try {
            return Double.parseDouble(text);
        } catch (NumberFormatException e) {
            return text;
        }
    }
}
