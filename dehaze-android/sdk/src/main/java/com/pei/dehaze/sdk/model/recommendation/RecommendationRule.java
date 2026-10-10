package com.pei.dehaze.sdk.model.recommendation;

import lombok.Data;

import java.util.List;

@Data
public class RecommendationRule {
    private Long id;
    private String ruleName;
    private String sceneType;
    private List<String> modelIds;
    private Integer weight;
    private Boolean enabled;
}
