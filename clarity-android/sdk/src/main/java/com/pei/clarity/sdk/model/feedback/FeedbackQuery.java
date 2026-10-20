package com.pei.clarity.sdk.model.feedback;

import com.pei.clarity.sdk.model.PageQuery;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@EqualsAndHashCode(callSuper = true)
public class FeedbackQuery extends PageQuery {
    private String keywords;
    private String feedbackType;
    private String status;
    private String relatedModule;
    private Integer priority;
    private Long assigneeId;
    private String startTime;
    private String endTime;
}
