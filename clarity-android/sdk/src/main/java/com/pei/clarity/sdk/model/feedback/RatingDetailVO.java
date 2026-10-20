package com.pei.clarity.sdk.model.feedback;

import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@EqualsAndHashCode(callSuper = true)
public class RatingDetailVO extends RatingPageVO {
    private String modelId;
}
