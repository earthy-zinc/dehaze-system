package com.pei.dehaze.model.query;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Size;
import lombok.Data;

/** 会话审计时间线查询参数（对齐 python {@code TimelineQuery} 的 {@code include} max_length=32） */
@Schema(description = "会话审计时间线查询参数")
@Data
public class AiObservabilityTimelineQuery {

    /** 包含项（默认含 raw 原始报文，传非 raw 值省略） */
    @Size(max = 32)
    private String include;
}
