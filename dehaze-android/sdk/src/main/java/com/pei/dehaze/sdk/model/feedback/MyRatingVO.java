package com.pei.dehaze.sdk.model.feedback;

import lombok.Data;

@Data
public class MyRatingVO {
    private Long id;
    private Long predLogId;
    private String algorithmName;
    private Integer rating;
    private String comment;
    private String[] tags;
    private String[] imageUrls;
    private Integer isAnonymous;
    private String adminReply;
    private String replyTime;
    private String createTime;
}
