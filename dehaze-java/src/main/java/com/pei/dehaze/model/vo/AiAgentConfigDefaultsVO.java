package com.pei.dehaze.model.vo;

import lombok.Data;

/**
 * Agent 推理参数系统默认值（代码常量 {@code AiAgentConfigResolver.REASONING_DEFAULTS} 的对外契约）。
 *
 * <p>Agent 配置表单「空值继承系统默认」的提示依赖本契约；字段与 dehaze-python
 * {@code AgentConfigDefaults} 逐项对应，前端不得硬编码默认值（会与代码常量漂移）。
 *
 * @author dehaze
 */
@Data
public class AiAgentConfigDefaultsVO {

    private Integer maxStepsReact;

    private Integer maxStepsPlan;

    private Integer maxStepsReflexion;

    private Integer maxIterationsReflexion;

    private Double reflexionThreshold;

    private Integer maxParallel;

    private Integer toolTimeout;

    private Integer tokenBudget;

    private Integer retryMax;
}
