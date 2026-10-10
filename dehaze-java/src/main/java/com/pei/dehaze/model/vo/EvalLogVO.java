package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.pei.dehaze.common.enums.LogStatusEnum;
import io.swagger.v3.oas.annotations.media.Schema;
import java.time.LocalDateTime;
import lombok.Data;

/**
 * 评估日志视图对象
 *
 * @author earthyzinc
 * @since 2024-06-12
 */
@Schema(description = "评估日志视图对象")
@Data
public class EvalLogVO {

    @Schema(description = "日志ID")
    private Long id;

    @Schema(description = "图像模型标识（统一模型规范 slug）")
    private String modelId;

    @Schema(description = "预测文件URL")
    private String predUrl;

    @Schema(description = "预测图内容MD5")
    private String predMd5;

    @Schema(description = "参考图片URL")
    private String gtUrl;

    @Schema(description = "参考图内容MD5")
    private String gtMd5;

    @Schema(description = "评估结果JSON")
    private String result;

    @Schema(description = "处理时间（秒）")
    private Integer time;

    @Schema(description = "任务状态：1=处理中,2=已完成,3=失败")
    private LogStatusEnum status;

    @Schema(description = "失败错误信息")
    private String errorMessage;

    @Schema(description = "创建时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;
}
