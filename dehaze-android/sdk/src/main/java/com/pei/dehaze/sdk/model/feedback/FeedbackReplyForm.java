package com.pei.dehaze.sdk.model.feedback;

import lombok.Data;

@Data
public class FeedbackReplyForm {
    private String content;
    private String replyType;
    private String[] attachments;
}
