package com.pei.clarity.model.vo;

import static org.assertj.core.api.Assertions.assertThat;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.pei.clarity.model.dto.ai.AiContract;
import java.util.List;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * 消息契约 JSON 形状：全局 non_null 下，嵌套 record 的可空字段仍显式下发 null。
 *
 * <p>prod 的 {@code default-property-inclusion: non_null} 只作用于未声明包含规则的属性，
 * {@code AiMessageVO} 的类级 ALWAYS 不会级联到 record 属性；record 自身也必须标注 ALWAYS，
 * 否则同一行数据 Java 下发 {@code {id,kind,seq,text}}，Python/Go 还带
 * {@code durationMs:null,tool:null}，三端键集合不一致。
 */
@DisplayName("消息契约 JSON：嵌套 record 的 null 键")
class AiMessageVoJsonIncludeTest {

    /** 复刻 application-prod.yml 的 default-property-inclusion: non_null */
    private final ObjectMapper mapper = new ObjectMapper().setSerializationInclusion(JsonInclude.Include.NON_NULL);

    @Test
    @DisplayName("block：durationMs/tool 为 null 时仍下发键")
    void blockKeepsNullKeys() throws Exception {
        JsonNode json =
                mapper.readTree(mapper.writeValueAsString(new AiContract.Block("b1", "text", 0L, "你好", null, null)));

        assertThat(json.has("durationMs")).isTrue();
        assertThat(json.get("durationMs").isNull()).isTrue();
        assertThat(json.has("tool")).isTrue();
        assertThat(json.get("tool").isNull()).isTrue();
    }

    @Test
    @DisplayName("interrupt/usage：可空字段与 subAgents 为 null 时仍下发键")
    void interruptAndUsageKeepNullKeys() throws Exception {
        AiContract.Interrupt interrupt = new AiContract.Interrupt(
                "int-1",
                "confirm",
                null,
                "predict",
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                "调用预测服务前需授权",
                "2026-10-21T10:00:00",
                "2026-10-21T10:30:00",
                true);
        JsonNode interruptJson = mapper.readTree(mapper.writeValueAsString(interrupt));

        assertThat(interruptJson.has("confirmKind")).isTrue();
        assertThat(interruptJson.has("recommendation")).isTrue();
        assertThat(interruptJson.has("usedDaily")).isTrue();
        assertThat(interruptJson.has("plan")).isTrue();

        JsonNode usageJson = mapper.readTree(mapper.writeValueAsString(new AiContract.Usage(1L, 2L, 3L, 4L, null)));
        assertThat(usageJson.has("subAgents")).isTrue();
        assertThat(usageJson.get("subAgents").isNull()).isTrue();
    }

    @Test
    @DisplayName("消息 VO：blocks 内的 block 与 usage 同样保留 null 键")
    void messageBlocksKeepNullKeys() throws Exception {
        AiMessageVO vo = new AiMessageVO();
        vo.setId(1L);
        vo.setConversationId(2L);
        vo.setRole("assistant");
        vo.setStatus("completed");
        vo.setBlocks(List.of(new AiContract.Block("b1", "text", 0L, "hi", null, null)));
        vo.setUsage(new AiContract.Usage(1L, 2L, 3L, 4L, null));
        vo.setArtifactIds(List.of());
        vo.setMemoryIds(List.of());
        vo.setSuggestions(List.of());

        JsonNode json = mapper.readTree(mapper.writeValueAsString(vo));

        assertThat(json.has("model")).isTrue();
        assertThat(json.has("pendingInterrupt")).isTrue();
        JsonNode block = json.get("blocks").get(0);
        assertThat(block.has("durationMs")).isTrue();
        assertThat(block.has("tool")).isTrue();
        assertThat(json.get("usage").has("subAgents")).isTrue();
    }
}
