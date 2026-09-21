package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.FieldFill;
import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import lombok.Data;

import java.math.BigDecimal;
import java.time.LocalDateTime;

/**
 * AI 计费记录（表 sys_ai_billing，只追加不删除）。
 *
 * <p>Python 推理链路为主要写入方；Java 侧承担管理端查询与对账。
 * 表无 update_time/create_by，故不继承 BaseEntity。
 */
@Data
public class SysAiBilling {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long userId;

    private Long conversationId;

    private Long messageId;

    private String requestId;

    private Long providerId;

    /** 实际使用模型标识（降级场景为降级模型） */
    private String model;

    /** 用户原选模型标识（NULL 表示未降级） */
    private String actualModel;

    private String errorCode;

    private Integer latencyMs;

    /** chat/chat_subagent/tool_llm/kb_inject/embedding/rerank/asr/tts */
    private String billType;

    private Integer inputTokens;

    private Integer cachedInputTokens;

    private Integer outputTokens;

    private Integer credits;

    private Integer creditsSaved;

    private Integer toolCredits;

    private Integer quotaConsumed;

    private Integer preDeduct;

    private BigDecimal cost;

    @TableField(fill = FieldFill.INSERT)
    private LocalDateTime createTime;
}
