package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.pei.dehaze.common.enums.LogStatusEnum;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.Map;

@Schema(description = "评估指标历史视图对象")
@Data
public class EvalMetricsVO {

    @Schema(description = "日志ID")
    private Long id;

    @Schema(description = "算法ID")
    private Long algorithmId;

    @Schema(description = "算法名称")
    private String algorithmName;

    @Schema(description = "预测文件URL")
    private String predUrl;

    @Schema(description = "参考图片URL")
    private String gtUrl;

    @Schema(description = "评估指标结果（Map）")
    private Map<String, Double> metrics;

    @Schema(description = "处理时间（毫秒）")
    private Integer time;

    @Schema(description = "任务状态")
    private LogStatusEnum status;

    @Schema(description = "失败错误信息")
    private String errorMessage;

    @Schema(description = "创建时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;
}
