package com.pei.dehaze.sdk.model.feedback;

import lombok.Data;

@Data
public class FeedbackCreateForm {
    private String feedbackType;
    private String title;
    private String content;
    private String contact;
    private String[] images;
    private String relatedModule;
}
