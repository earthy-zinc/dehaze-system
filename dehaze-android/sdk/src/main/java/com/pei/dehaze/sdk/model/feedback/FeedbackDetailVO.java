package com.pei.dehaze.sdk.model.feedback;

import lombok.Data;
import lombok.EqualsAndHashCode;

import java.util.List;

@Data
@EqualsAndHashCode(callSuper = true)
public class FeedbackDetailVO extends FeedbackPageVO {
    private String contact;
    private String[] images;
    private String assignedTime;
    private String closeReason;
    private List<FeedbackReplyVO> replies;
}
