package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.util.List;
import java.util.Map;

@Data
@Schema(description = "反馈统计VO")
public class FeedbackStatsVO {

    @Schema(description = "反馈总数")
    private Long totalFeedback;

    @Schema(description = "类型分布")
    private Map<String, Long> typeDistribution;

    @Schema(description = "模块分布")
    private List<ModuleCount> moduleDistribution;

    @Schema(description = "状态分布")
    private Map<String, Long> statusDistribution;

    @Schema(description = "平均响应时间（毫秒）")
    private Long averageResponseTime;

    @Schema(description = "平均关闭时间（毫秒）")
    private Long averageCloseTime;

    @Schema(description = "高频关键词")
    private List<KeywordCount> topKeywords;

    @Data
    @Schema(description = "模块统计项")
    public static class ModuleCount {
        private String module;
        private Long count;
    }

    @Data
    @Schema(description = "关键词统计项")
    public static class KeywordCount {
        private String keyword;
        private Long count;
    }
}
