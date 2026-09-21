package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.FieldFill;
import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import lombok.Data;

import java.time.LocalDateTime;

/** AI 对话每次 LLM 调用明细（表 sys_ai_llm_call，只追加） */
@Data
public class SysAiLlmCall {

    @TableId(type = IdType.AUTO)
    private Long id;

    private String traceId;

    private Integer seq;

    private Integer stepPosition;

    private String model;

    private LocalDateTime startTime;

    /** 1:成功;2:失败;3:超时 */
    private Integer status;

    private String errorType;

    private Integer durationMs;

    private Integer firstTokenMs;

    private Integer promptTokens;

    private Integer completionTokens;

    private Integer cachedTokens;

    private String toolCall;

    private String inputSnapshot;

    private String outputSnapshot;

    private String attempts;

    private String rawRequest;

    private String rawResponse;

    @TableField(fill = FieldFill.INSERT)
    private LocalDateTime createTime;
}
