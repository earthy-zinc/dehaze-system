package com.pei.dehaze.sdk.model.feedback;

import com.pei.dehaze.sdk.model.PageQuery;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@EqualsAndHashCode(callSuper = true)
public class RatingQuery extends PageQuery {
    private String keywords;
    private Long algorithmId;
    private Integer ratingMin;
    private Integer ratingMax;
    private Boolean hasComment;
    private String[] tags;
    private String startTime;
    private String endTime;
}
