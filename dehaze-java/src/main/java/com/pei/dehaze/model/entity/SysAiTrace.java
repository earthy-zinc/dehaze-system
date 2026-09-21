package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.FieldFill;
import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import lombok.Data;

import java.time.LocalDateTime;

/** AI 对话过程链汇总记录（表 sys_ai_trace，只追加，保留 180 天） */
@Data
public class SysAiTrace {

    @TableId(type = IdType.AUTO)
    private Long id;

    private String traceId;

    private Long conversationId;

    private Long messageId;

    private String agentCode;

    /** conversation/summary/memory_extraction/suggestion/step_summary */
    private String traceType;

    private String model;

    /** 1:成功;2:失败;3:中断;4:超时 */
    private Integer status;

    private String errorType;

    private Integer durationMs;

    private Integer firstTokenMs;

    private Integer llmCallCount;

    private Integer totalTokens;

    private Integer promptTokens;

    private Integer completionTokens;

    private Integer cachedTokens;

    private Integer stepCount;

    private String contextSnapshot;

    private String errorDetail;

    @TableField(fill = FieldFill.INSERT)
    private LocalDateTime createTime;
}
