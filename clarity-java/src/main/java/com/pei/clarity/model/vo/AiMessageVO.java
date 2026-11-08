package com.pei.clarity.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.pei.clarity.model.dto.ai.AiContract;
import java.time.LocalDateTime;
import java.util.List;
import lombok.Data;

/**
 * AI 消息响应（契约 {@code Message} 形状）。
 *
 * <p>正文、思考、工具调用统一由 {@code blocks} 承载；计划与待恢复中断随消息返回；产物与记忆只返回引用 ID。
 *
 * <p>类级 {@code ALWAYS}：契约里可空字段（model/error/plan/pendingInterrupt 等）以显式 null 返回，
 * 不被全局 {@code default-property-inclusion: non_null} 省略，保证三端响应的键集合一致。
 *
 * @author clarity
 */
@JsonInclude(JsonInclude.Include.ALWAYS)
@Data
public class AiMessageVO {

    private Long id;

    private Long conversationId;

    private Long parentMessageId;

    /** 取值: system | user | assistant | tool */
    private String role;

    /** 取值: running | completed | failed | canceled | awaiting_input */
    private String status;

    private String model;

    private String error;

    private List<AiContract.Block> blocks;

    private AiContract.Plan plan;

    private AiContract.Interrupt pendingInterrupt;

    private AiContract.Usage usage;

    private List<Long> artifactIds;

    private List<Long> memoryIds;

    private List<String> suggestions;

    private boolean edited;

    /** 同父 assistant 兄弟中的序号（1 起）；self 视角列表与消息详情填充，root/非 assistant 为 null */
    private Integer siblingIndex;

    /** 同父 assistant 兄弟总数；self 视角列表与消息详情填充，root/非 assistant 为 null */
    private Integer siblingCount;

    @JsonFormat(pattern = "yyyy-MM-dd'T'HH:mm:ss")
    private LocalDateTime createTime;

    @JsonFormat(pattern = "yyyy-MM-dd'T'HH:mm:ss")
    private LocalDateTime updateTime;
}
