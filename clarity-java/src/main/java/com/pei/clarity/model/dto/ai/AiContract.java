package com.pei.clarity.model.dto.ai;

import com.fasterxml.jackson.annotation.JsonInclude;
import java.util.List;
import java.util.Map;

/**
 * GENERATED FROM config/schema/ai-conversation.schema.json — DO NOT EDIT.
 * 由 scripts/tooling/gen-ai-contract.mjs 生成；改契约请改 Schema 后重新运行生成器。
 * 枚举字段一律为 String，取值见字段上方注释；时间字段一律为 String（ISO-8601）。
 * 每个 record 标注 @JsonInclude(ALWAYS)：嵌套结构的显式 null（durationMs/tool 等）不被
 * 全局 default-property-inclusion=non_null 省略，保证三端响应的键集合一致。
 */
public final class AiContract {

    private AiContract() {
    }

    @JsonInclude(JsonInclude.Include.ALWAYS)
    public record ToolCall(
            String callId,
            String name,
            Map<String, Object> input,
            String output,
            String summary,
            // 取值: running | success | failed | skipped
            String status,
            String error,
            Long latencyMs,
            String agentCode,
            boolean subAgent
    ) {
    }

    @JsonInclude(JsonInclude.Include.ALWAYS)
    public record Block(
            String id,
            // 取值: text | thinking | tool_use
            String kind,
            long seq,
            String text,
            Long durationMs,
            ToolCall tool
    ) {
    }

    @JsonInclude(JsonInclude.Include.ALWAYS)
    public record PlanTask(
            String id,
            String description,
            List<String> dependsOn,
            // 取值: pending | running | completed | failed
            String status,
            String paradigm
    ) {
    }

    @JsonInclude(JsonInclude.Include.ALWAYS)
    public record PlanRevision(
            long revisionNo,
            String reason,
            String actor,
            String at
    ) {
    }

    @JsonInclude(JsonInclude.Include.ALWAYS)
    public record Plan(
            String phase,
            // 取值: pending | running | completed | failed
            String status,
            List<PlanTask> tasks,
            List<PlanRevision> revisions,
            boolean awaitingApproval
    ) {
    }

    // 允许附加字段（Schema additionalProperties=true），实现侧自行透传
    @JsonInclude(JsonInclude.Include.ALWAYS)
    public record Recommendation(
            String modelId,
            String modelName,
            String reason
    ) {
    }

    // 允许附加字段（Schema additionalProperties=true），实现侧自行透传
    @JsonInclude(JsonInclude.Include.ALWAYS)
    public record Alternative(
            String modelId,
            String modelName,
            Double matchScore,
            String reason
    ) {
    }

    @JsonInclude(JsonInclude.Include.ALWAYS)
    public record Interrupt(
            String id,
            // 取值: confirm | quota | async_wait | plan_approve
            String type,
            // 取值: algorithm_recommend | tool_permission | dangerous_op | write_conflict
            String confirmKind,
            String action,
            Recommendation recommendation,
            List<Alternative> alternatives,
            String upgradeTip,
            Long usedDaily,
            Long dailyLimit,
            Long usedMonthly,
            Long monthlyLimit,
            String estDuration,
            Long imageCount,
            List<PlanTask> plan,
            String reason,
            String detail,
            String createdAt,
            String expiresAt,
            boolean actionable
    ) {
    }

    @JsonInclude(JsonInclude.Include.ALWAYS)
    public record SubAgentUsage(
            String agentCode,
            long inputTokens,
            long outputTokens,
            long cachedInputTokens,
            long credits
    ) {
    }

    @JsonInclude(JsonInclude.Include.ALWAYS)
    public record Usage(
            long inputTokens,
            long outputTokens,
            long cachedInputTokens,
            long credits,
            List<SubAgentUsage> subAgents
    ) {
    }

    @JsonInclude(JsonInclude.Include.ALWAYS)
    public record Message(
            long id,
            long conversationId,
            Long parentMessageId,
            // 取值: system | user | assistant | tool
            String role,
            // 取值: running | completed | failed | canceled | awaiting_input
            String status,
            String model,
            String error,
            List<Block> blocks,
            Plan plan,
            Interrupt pendingInterrupt,
            Usage usage,
            List<Long> artifactIds,
            List<Long> memoryIds,
            List<String> suggestions,
            boolean edited,
            Long siblingIndex,
            Long siblingCount,
            String createTime,
            String updateTime
    ) {
    }

    @JsonInclude(JsonInclude.Include.ALWAYS)
    public record MessageList(
            List<Message> list,
            long total,
            boolean hasMore
    ) {
    }

    @JsonInclude(JsonInclude.Include.ALWAYS)
    public record MessageSend(
            String content,
            String model
    ) {
    }

    @JsonInclude(JsonInclude.Include.ALWAYS)
    public record MessageResume(
            Boolean confirm,
            Map<String, Object> params,
            Map<String, Object> planEdit
    ) {
    }

    @JsonInclude(JsonInclude.Include.ALWAYS)
    public record Artifact(
            long id,
            long conversationId,
            long messageId,
            String type,
            String refType,
            Long refId,
            Map<String, Object> summary,
            boolean isInvalid,
            String createTime
    ) {
    }

    @JsonInclude(JsonInclude.Include.ALWAYS)
    public record Feedback(
            Long id,
            long messageId,
            long userId,
            Long conversationId,
            // 取值: 1 | -1
            long rating,
            List<String> tags,
            String comment,
            String createTime,
            String updateTime
    ) {
    }

    @JsonInclude(JsonInclude.Include.ALWAYS)
    public record MessageStartEvent(
            String streamSessionId,
            Message userMessage,
            Message assistantMessage
    ) {
    }

    @JsonInclude(JsonInclude.Include.ALWAYS)
    public record BlockStartEvent(
            long messageId,
            Block block
    ) {
    }

    @JsonInclude(JsonInclude.Include.ALWAYS)
    public record BlockDeltaEvent(
            long messageId,
            String blockId,
            String delta
    ) {
    }

    @JsonInclude(JsonInclude.Include.ALWAYS)
    public record BlockStopEvent(
            long messageId,
            Block block
    ) {
    }

    @JsonInclude(JsonInclude.Include.ALWAYS)
    public record PlanUpdateEvent(
            long messageId,
            Plan plan
    ) {
    }

    @JsonInclude(JsonInclude.Include.ALWAYS)
    public record InterruptEvent(
            long messageId,
            Interrupt interrupt
    ) {
    }

    @JsonInclude(JsonInclude.Include.ALWAYS)
    public record UsageEvent(
            long messageId,
            Usage usage
    ) {
    }

    @JsonInclude(JsonInclude.Include.ALWAYS)
    public record SuggestionsEvent(
            long messageId,
            List<String> suggestions
    ) {
    }

    @JsonInclude(JsonInclude.Include.ALWAYS)
    public record MessageEndEvent(
            long messageId,
            // 取值: running | completed | failed | canceled | awaiting_input
            String status,
            String error,
            Usage usage
    ) {
    }

    @JsonInclude(JsonInclude.Include.ALWAYS)
    public record ErrorEvent(
            String code,
            String message,
            Long messageId
    ) {
    }
}
