package com.pei.dehaze.sdk.model.feedback;

import lombok.Data;

@Data
public class FeedbackPageVO {
    private Long id;
    private Long userId;
    private String username;
    private String feedbackType;
    private String title;
    private String content;
    private String status;
    private Integer priority;
    private Long assigneeId;
    private String assigneeName;
    private String relatedModule;
    private String[] tags;
    private String createTime;
    private String updateTime;
}
