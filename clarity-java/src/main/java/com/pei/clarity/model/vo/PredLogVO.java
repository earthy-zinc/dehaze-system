package com.pei.clarity.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.pei.clarity.common.enums.LogStatusEnum;
import io.swagger.v3.oas.annotations.media.Schema;
import java.time.LocalDateTime;
import lombok.Data;

/**
 * 图像任务历史项视图对象（GET /api/v1/images/tasks）
 */
@Schema(description = "图像任务历史项视图对象")
@Data
public class PredLogVO {

    @Schema(description = "任务ID")
    private Long id;

    @Schema(description = "图像模型标识（统一模型规范 slug）")
    private String model;

    @Schema(description = "原图MD5")
    private String originMd5;

    @Schema(description = "原始图片URL")
    private String originUrl;

    @Schema(description = "结果图MD5")
    private String predMd5;

    @Schema(description = "处理结果URL")
    private String predUrl;

    @Schema(description = "任务状态：1=处理中,2=已完成,3=失败,4=已取消")
    private LogStatusEnum status;

    @Schema(description = "失败错误信息")
    private String errorMessage;

    @Schema(description = "推理耗时（秒）")
    private Integer time;

    @Schema(description = "创建时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;
}
