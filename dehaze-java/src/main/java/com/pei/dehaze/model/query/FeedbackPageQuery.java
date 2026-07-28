package com.pei.dehaze.model.query;

import com.pei.dehaze.common.base.BasePageQuery;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;

import java.time.LocalDateTime;

@Data
@EqualsAndHashCode(callSuper = true)
@Schema(description = "反馈后台分页查询参数")
public class FeedbackPageQuery extends BasePageQuery {

    @Schema(description = "标题/内容关键字")
    private String keywords;

    @Schema(description = "反馈类型(suggestion/bug/experience/complaint)")
    private String feedbackType;

    @Schema(description = "反馈状态(pending/processing/replied/closed)")
    private String status;

    @Schema(description = "相关模块")
    private String relatedModule;

    @Schema(description = "优先级(1:普通;2:紧急;3:高优)")
    private Integer priority;

    @Schema(description = "处理人ID")
    private Long assigneeId;

    @Schema(description = "起始时间")
    private LocalDateTime startTime;

    @Schema(description = "结束时间")
    private LocalDateTime endTime;
}
