package com.pei.dehaze.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.controller.AiAgentController;
import com.pei.dehaze.mapper.AiInsightMapper;
import com.pei.dehaze.model.vo.AiAgentConfigDefaultsVO;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * 推理参数系统默认值对外契约单测。
 *
 * <p>字段与取值是 java / python 共用契约（python 侧 {@code AgentConfigDefaults}）：前端不得硬编码，
 * 常量调整时本契约必须同步，否则前端展示的"空值继承默认"会与实际推理参数漂移。
 */
@DisplayName("Agent 推理参数默认值对外契约")
@ExtendWith(MockitoExtension.class)
class AiAgentConfigDefaultsTest {

    @Mock
    private AiInsightMapper insightMapper;

    private AiAgentConfigResolver resolver;

    @BeforeEach
    void setUp() {
        resolver = new AiAgentConfigResolver(insightMapper);
    }

    @Test
    @DisplayName("9 项默认值与 python REASONING_DEFAULTS 逐项一致")
    void defaultsMatchPythonContract() {
        AiAgentConfigDefaultsVO defaults = resolver.reasoningDefaults();

        assertThat(defaults.getMaxStepsReact()).isEqualTo(20);
        assertThat(defaults.getMaxStepsPlan()).isEqualTo(30);
        assertThat(defaults.getMaxStepsReflexion()).isEqualTo(15);
        assertThat(defaults.getMaxIterationsReflexion()).isEqualTo(3);
        assertThat(defaults.getReflexionThreshold()).isEqualTo(0.8);
        assertThat(defaults.getMaxParallel()).isEqualTo(5);
        assertThat(defaults.getToolTimeout()).isEqualTo(60);
        assertThat(defaults.getTokenBudget()).isEqualTo(500000);
        assertThat(defaults.getRetryMax()).isEqualTo(2);
    }

    @Test
    @DisplayName("序列化键名为 camelCase 且字段无遗漏（漏映射会输出 null 键值对）")
    void jsonKeysAreCamelCaseWithoutNullFields() throws Exception {
        String json = new ObjectMapper().writeValueAsString(resolver.reasoningDefaults());
        JsonNode node = new ObjectMapper().readTree(json);

        List<String> keys = new java.util.ArrayList<>();
        for (Iterator<String> it = node.fieldNames(); it.hasNext(); ) {
            keys.add(it.next());
        }
        assertThat(keys).containsExactlyInAnyOrder(
                "maxStepsReact", "maxStepsPlan", "maxStepsReflexion", "maxIterationsReflexion",
                "reflexionThreshold", "maxParallel", "toolTimeout", "tokenBudget", "retryMax");
        assertThat(node.findValues("null")).isEmpty();
        assertThat(node.findValues("")).isEmpty();
    }

    @Test
    @DisplayName("控制器以成功信封返回默认值")
    void controllerWrapsDefaultsInSuccessEnvelope() {
        Result<AiAgentConfigDefaultsVO> result = new AiAgentController(null, null, resolver).configDefaults();

        assertThat(result.getData().getMaxStepsReact()).isEqualTo(20);
    }
}
