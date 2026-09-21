package com.pei.dehaze.model.query;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Size;
import jakarta.validation.constraints.Min;
import lombok.Data;

/** 过程链检索/导出查询参数（空 time 串表示不过滤） */
@Schema(description = "过程链检索查询参数")
@Data
public class AiObservabilityTraceQuery {

    @Min(1)
    private Integer pageNum = 1;

    @Min(1)
    @Max(100)
    private Integer pageSize = 10;

    private Long conversationId;

    /** 用户归属筛选（经会话表关联） */
    private Long userId;

    /** 1:成功;2:失败;3:中断;4:超时 */
    @Min(1)
    @Max(4)
    private Integer status;

    @Size(max = 64)
    private String agentCode;

    @Size(max = 64)
    private String model;

    @Size(max = 32)
    private String errorType;

    /** 匹配 trace_id 或会话标题模糊 */
    @Size(max = 64)
    private String keyword;

    /** memory/kb/tools */
    private String capability;

    private String startTime;

    private String endTime;
}
