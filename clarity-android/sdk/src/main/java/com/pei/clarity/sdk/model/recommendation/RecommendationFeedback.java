package com.pei.clarity.sdk.model.recommendation;

import lombok.Data;

@Data
public class RecommendationFeedback {
    private Long recommendationId;
    private Boolean useful;
}
