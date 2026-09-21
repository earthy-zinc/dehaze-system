package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.util.ArrayList;
import java.util.List;

/** 资源消耗聚合：按维度分页聚合 + 按日 Token 趋势 */
@Schema(description = "资源消耗聚合")
@Data
public class AiObservabilityCostsVO {

    private List<Item> items = new ArrayList<>();

    /** 聚合分组总数 */
    private Long total;

    private List<TrendItem> trend = new ArrayList<>();

    @Schema(description = "按维度聚合行")
    @Data
    public static class Item {

        private String model;

        private String agentCode;

        private Long userId;

        private Long traceCount;

        private Long totalTokens;

        private Long promptTokens;

        private Long completionTokens;

        private Long cachedTokens;
    }

    @Schema(description = "按日 Token 消耗趋势")
    @Data
    public static class TrendItem {

        private String date;

        private Long traceCount;

        private Long totalTokens;

        private Long promptTokens;

        private Long completionTokens;

        private Long cachedTokens;
    }
}
