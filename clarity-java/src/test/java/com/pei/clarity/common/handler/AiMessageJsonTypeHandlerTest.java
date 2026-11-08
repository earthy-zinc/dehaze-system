package com.pei.clarity.common.handler;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.pei.clarity.model.dto.ai.AiContract;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.util.List;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

/**
 * 消息 JSON 列处理器单测：契约 DTO 的读写形状。
 *
 * <p>MyBatis-Plus 自带 {@code JacksonTypeHandler} 因泛型擦除会把 {@code Block[]} 读成 {@code List<Map>}，
 * 本用例断言按 {@link AiContract} 的 record 类型落位（thinking 的 durationMs、tool_use 的 tool 子结构），
 * 并断言回写仍是契约的 camelCase 键，供 Python/前端互认。
 */
@DisplayName("消息 JSON 列处理器（blocks/plan/pending_interrupt）")
class AiMessageJsonTypeHandlerTest {

    private static final String BLOCKS_JSON = "["
            + "{\"id\":\"b1\",\"kind\":\"thinking\",\"seq\":0,\"text\":\"先看图\",\"durationMs\":120},"
            + "{\"id\":\"b2\",\"kind\":\"text\",\"seq\":0,\"text\":\"已完成\"},"
            + "{\"id\":\"b3\",\"kind\":\"tool_use\",\"seq\":1,\"tool\":{"
            + "\"callId\":\"call-1\",\"name\":\"predict\",\"input\":{\"imageId\":9},"
            + "\"output\":\"ok\",\"summary\":\"去雾\",\"status\":\"success\","
            + "\"latencyMs\":30,\"agentCode\":\"vision\",\"subAgent\":true}}]";

    @Test
    @DisplayName("blocks：按 Block/ToolCall 契约反序列化，不被泛型擦除退化为 Map")
    void blocksDeserializeToContractRecords() throws Exception {
        ResultSet rs = mock(ResultSet.class);
        when(rs.getString("blocks")).thenReturn(BLOCKS_JSON);

        List<AiContract.Block> blocks = new AiBlockListTypeHandler().getNullableResult(rs, "blocks");

        assertThat(blocks).hasSize(3);
        assertThat(blocks).extracting(AiContract.Block::kind).containsExactly("thinking", "text", "tool_use");
        assertThat(blocks.get(0).durationMs()).isEqualTo(120L);
        assertThat(blocks.get(1).text()).isEqualTo("已完成");
        assertThat(blocks.get(2).tool()).isNotNull();
        assertThat(blocks.get(2).tool().name()).isEqualTo("predict");
        assertThat(blocks.get(2).tool().status()).isEqualTo("success");
        assertThat(blocks.get(2).tool().input()).containsEntry("imageId", 9);
        assertThat(blocks.get(2).tool().subAgent()).isTrue();
    }

    @Test
    @DisplayName("plan：phase/status/tasks/revisions/awaitingApproval 全量落位")
    void planDeserialize() throws Exception {
        String json = "{\"phase\":\"analyze\",\"status\":\"running\","
                + "\"tasks\":[{\"id\":\"t1\",\"description\":\"拆解\",\"dependsOn\":[\"t0\"],\"status\":\"completed\"}],"
                + "\"revisions\":[{\"revisionNo\":2,\"reason\":\"补充\",\"actor\":\"user\",\"at\":\"2026-10-21T10:00:00\"}],"
                + "\"awaitingApproval\":true}";
        ResultSet rs = mock(ResultSet.class);
        when(rs.getString("plan")).thenReturn(json);

        AiContract.Plan plan = new AiPlanTypeHandler().getNullableResult(rs, "plan");

        assertThat(plan.status()).isEqualTo("running");
        assertThat(plan.awaitingApproval()).isTrue();
        assertThat(plan.tasks()).hasSize(1);
        assertThat(plan.tasks().get(0).dependsOn()).containsExactly("t0");
        assertThat(plan.revisions().get(0).revisionNo()).isEqualTo(2L);
        assertThat(plan.revisions().get(0).actor()).isEqualTo("user");
    }

    @Test
    @DisplayName("pending_interrupt：可空字段与 expiresAt/actionable 落位")
    void pendingInterruptDeserialize() throws Exception {
        String json = "{\"id\":\"int-1\",\"type\":\"quota\",\"usedDaily\":5,\"dailyLimit\":5,"
                + "\"recommendation\":{\"modelId\":\"m1\",\"modelName\":\"模型一\",\"reason\":\"更省\"},"
                + "\"createdAt\":\"2026-10-21T10:00:00\",\"expiresAt\":\"2026-10-21T10:30:00\",\"actionable\":true}";
        ResultSet rs = mock(ResultSet.class);
        when(rs.getString("pending_interrupt")).thenReturn(json);

        AiContract.Interrupt interrupt = new AiInterruptTypeHandler().getNullableResult(rs, "pending_interrupt");

        assertThat(interrupt.type()).isEqualTo("quota");
        assertThat(interrupt.usedDaily()).isEqualTo(5L);
        assertThat(interrupt.recommendation().modelName()).isEqualTo("模型一");
        assertThat(interrupt.expiresAt()).isEqualTo("2026-10-21T10:30:00");
        assertThat(interrupt.actionable()).isTrue();
        assertThat(interrupt.confirmKind()).isNull();
    }

    @Test
    @DisplayName("blocks：回写使用契约 camelCase 键（Python/前端可互认）")
    void blocksSerializeWithContractKeys() throws Exception {
        List<AiContract.Block> blocks = List.of(new AiContract.Block("b1", "text", 0L, "你好", 80L, null));
        PreparedStatement ps = mock(PreparedStatement.class);

        new AiBlockListTypeHandler().setNonNullParameter(ps, 1, blocks, null);

        ArgumentCaptor<String> json = ArgumentCaptor.forClass(String.class);
        verify(ps).setString(eq(1), json.capture());
        assertThat(json.getValue())
                .contains("\"durationMs\":80")
                .contains("\"kind\":\"text\"")
                .doesNotContain("duration_ms");
    }

    @Test
    @DisplayName("空列：返回 null（对应可空 JSON 列）")
    void blankColumnReturnsNull() throws Exception {
        ResultSet rs = mock(ResultSet.class);
        when(rs.getString("plan")).thenReturn(null);

        assertThat(new AiPlanTypeHandler().getNullableResult(rs, "plan")).isNull();
    }
}
