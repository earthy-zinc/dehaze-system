package com.pei.dehaze.model.vo;

import com.pei.dehaze.common.enums.LogStatusEnum;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

@Schema(description = "对比报告结果视图对象")
@Data
public class CompareReportResultVO {

    @Schema(description = "异步任务ID")
    private Long taskId;

    @Schema(description = "任务状态：1=处理中,2=已完成,3=失败")
    private LogStatusEnum status;

    @Schema(description = "下载地址（任务完成后返回）")
    private String downloadUrl;

    @Schema(description = "失败错误信息（任务失败时返回）")
    private String errorMessage;
}
