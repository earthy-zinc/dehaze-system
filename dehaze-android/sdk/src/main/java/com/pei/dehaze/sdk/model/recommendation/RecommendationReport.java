package com.pei.dehaze.sdk.model.recommendation;

import lombok.Data;

import java.util.List;

@Data
public class RecommendationReport {
    private Integer totalRecommendations;
    private Double adoptionRate;
    private Double satisfactionRate;
    private Double coverageRate;
    private Double coldStartSuccessRate;
    private List<TrendItem> trend;

    @Data
    public static class TrendItem {
        private String date;
        private Double adoptionRate;
    }
}
