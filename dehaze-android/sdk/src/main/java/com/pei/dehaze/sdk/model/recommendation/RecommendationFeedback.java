package com.pei.dehaze.sdk.model.recommendation;

import lombok.Data;

@Data
public class RecommendationFeedback {
    private Long recommendationId;
    private Boolean useful;
}
