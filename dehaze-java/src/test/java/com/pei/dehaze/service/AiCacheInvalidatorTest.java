package com.pei.dehaze.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.data.redis.core.StringRedisTemplate;

import java.util.Collection;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatCode;
import static org.mockito.ArgumentMatchers.anyCollection;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

/**
 * AI 域缓存失效器单测：键清单必须与 dehaze-python 的 ai_agent_service 键规范逐字一致，
 * 否则运行面（python）读到 java 写操作后的脏快照。
 */
@DisplayName("AiCacheInvalidator 跨端缓存键契约")
@ExtendWith(MockitoExtension.class)
class AiCacheInvalidatorTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    @Mock
    private StringRedisTemplate stringRedisTemplate;

    private AiCacheInvalidator invalidator;

    @BeforeEach
    void setUp() {
        invalidator = new AiCacheInvalidator(stringRedisTemplate);
    }

    @Test
    @DisplayName("Agent 写操作失效 python 键规范全量 6 键并逐键广播")
    void evictAgentCachesDeletesPythonKeySet() {
        invalidator.evictAgentCaches("default", 7L);

        ArgumentCaptor<Collection<String>> keysCaptor = ArgumentCaptor.forClass(Collection.class);
        verify(stringRedisTemplate).delete(keysCaptor.capture());
        assertThat(keysCaptor.getValue()).containsExactlyInAnyOrder(
                "ai:agent:default",
                "ai:agent:7:skills",
                "ai:agent:7:mcp",
                "ai:agent:7:subagents",
                "ai:agent:7:published",
                "ai:agent:list:enabled");
        verify(stringRedisTemplate, times(6)).convertAndSend(eq("cache:invalidation"), anyString());
    }

    @Test
    @DisplayName("发布/回滚只失效已发布版本键")
    void evictPublishedVersionOnlyTouchesPublishedKey() {
        invalidator.evictPublishedVersion(7L);

        ArgumentCaptor<Collection<String>> keysCaptor = ArgumentCaptor.forClass(Collection.class);
        verify(stringRedisTemplate).delete(keysCaptor.capture());
        assertThat(keysCaptor.getValue()).containsExactly("ai:agent:7:published");
    }

    @Test
    @DisplayName("广播载荷为 python 可识别的 {type:key,key,senderId} 格式，含发送实例标识")
    void broadcastPayloadMatchesPythonProtocol() throws Exception {
        invalidator.evict("ai:agent:list:enabled");

        ArgumentCaptor<String> payloadCaptor = ArgumentCaptor.forClass(String.class);
        verify(stringRedisTemplate).convertAndSend(eq("cache:invalidation"), payloadCaptor.capture());
        JsonNode payload = MAPPER.readTree(payloadCaptor.getValue());
        assertThat(payload.get("type").asText()).isEqualTo("key");
        assertThat(payload.get("key").asText()).isEqualTo("ai:agent:list:enabled");
        assertThat(payload.get("senderId").asText()).isNotBlank();
    }

    @Test
    @DisplayName("MCP 变更广播 {type:ai_graph_invalidate,senderId}：python 据此失效进程内推理图缓存")
    void reasoningGraphInvalidationPayloadMatchesProtocol() throws Exception {
        invalidator.evictReasoningGraphs();

        ArgumentCaptor<String> payloadCaptor = ArgumentCaptor.forClass(String.class);
        verify(stringRedisTemplate).convertAndSend(eq("cache:invalidation"), payloadCaptor.capture());
        JsonNode payload = MAPPER.readTree(payloadCaptor.getValue());
        assertThat(payload.get("type").asText()).isEqualTo("ai_graph_invalidate");
        assertThat(payload.get("senderId").asText()).isNotBlank();
        assertThat(payload.has("key")).isFalse();
    }

    @Test
    @DisplayName("Redis 故障不影响业务写入（失效失败只告警）")
    void redisFailureDoesNotPropagate() {
        doThrow(new RuntimeException("redis down")).when(stringRedisTemplate).delete(anyCollection());

        assertThatCode(() -> invalidator.evict("ai:agent:list:enabled")).doesNotThrowAnyException();
    }
}
