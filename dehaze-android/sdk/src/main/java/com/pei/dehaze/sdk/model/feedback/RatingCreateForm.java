package com.pei.dehaze.sdk.model.feedback;

import lombok.Data;

@Data
public class RatingCreateForm {
    private Long predLogId;
    private Integer rating;
    private String comment;
    private String[] tags;
    private String[] imageUrls;
    private Integer isAnonymous;
}
