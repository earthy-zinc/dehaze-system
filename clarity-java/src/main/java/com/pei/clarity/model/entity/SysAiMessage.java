package com.pei.clarity.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableLogic;
import com.baomidou.mybatisplus.annotation.TableName;
import com.pei.clarity.common.base.BaseEntity;
import com.pei.clarity.common.handler.AiBlockListTypeHandler;
import com.pei.clarity.common.handler.AiInterruptTypeHandler;
import com.pei.clarity.common.handler.AiPlanTypeHandler;
import com.pei.clarity.common.handler.AiSubAgentUsageListTypeHandler;
import com.pei.clarity.common.handler.LongListTypeHandler;
import com.pei.clarity.common.handler.StringListTypeHandler;
import com.pei.clarity.model.dto.ai.AiContract;
import java.io.Serial;
import java.util.List;
import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * AI 对话消息
 *
 * <p>消息只有一种形状：正文、思考、工具调用统一由 {@code blocks}（{@link AiContract.Block} 数组）承载，
 * 不再有独立的正文列；执行计划与待恢复中断随消息持久化，产物只存引用（{@code sys_ai_artifact}）。
 *
 * @author clarity
 */
@Data
@EqualsAndHashCode(callSuper = false)
@TableName(value = "sys_ai_message", autoResultMap = true)
public class SysAiMessage extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long conversationId;

    private Long parentMessageId;

    private String role;

    /** 内容块（kind 取 text/thinking/tool_use，seq 为同 kind 分区内序号） */
    @TableField(typeHandler = AiBlockListTypeHandler.class)
    private List<AiContract.Block> blocks;

    /** 执行计划（含 tasks/revisions/awaitingApproval） */
    @TableField(typeHandler = AiPlanTypeHandler.class)
    private AiContract.Plan plan;

    /** 待恢复中断（含 expiresAt/actionable） */
    @TableField(typeHandler = AiInterruptTypeHandler.class)
    private AiContract.Interrupt pendingInterrupt;

    /** 推荐追问列表 */
    @TableField(typeHandler = StringListTypeHandler.class)
    private List<String> suggestions;

    private String model;

    /** 本轮通道状态：running/completed/failed/canceled/awaiting_input */
    private String status;

    private String error;

    private Integer inputTokens;

    private Integer outputTokens;

    private Integer cachedInputTokens;

    private Long credits;

    private String taskId;

    @TableField(typeHandler = LongListTypeHandler.class)
    private List<Long> usedMemoryIds;

    /** 子智能体分项用量（与 usage.subAgents 同形，收尾时落库） */
    @TableField(typeHandler = AiSubAgentUsageListTypeHandler.class)
    private List<AiContract.SubAgentUsage> subAgentUsage;

    private Integer edited;

    @TableLogic(value = "0", delval = "id")
    private Long deleted;

    @Serial
    private static final long serialVersionUID = 1L;
}
