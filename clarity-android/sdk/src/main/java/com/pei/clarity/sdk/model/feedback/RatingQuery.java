package com.pei.clarity.sdk.model.feedback;

import com.pei.clarity.sdk.model.PageQuery;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@EqualsAndHashCode(callSuper = true)
public class RatingQuery extends PageQuery {
    private String keywords;
    private String modelId;
    private Integer ratingMin;
    private Integer ratingMax;
    private Boolean hasComment;
    private String[] tags;
    private String startTime;
    private String endTime;
}
