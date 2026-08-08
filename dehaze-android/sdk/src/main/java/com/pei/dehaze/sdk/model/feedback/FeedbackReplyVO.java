package com.pei.dehaze.sdk.model.feedback;

import lombok.Data;

@Data
public class FeedbackReplyVO {
    private Long id;
    private Long feedbackId;
    private Long replierId;
    private String replierName;
    private Integer replierType;
    private String content;
    private String replyType;
    private String[] attachments;
    private String createTime;
}
