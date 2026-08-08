package com.pei.dehaze.sdk.model.feedback;

import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@EqualsAndHashCode(callSuper = true)
public class RatingDetailVO extends RatingPageVO {
    private Long algorithmId;
}
