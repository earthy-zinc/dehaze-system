package com.pei.dehaze.module.ai.controller.admin.workflow.vo;

import com.pei.dehaze.framework.common.enums.CommonStatusEnum;
import com.pei.dehaze.framework.common.pojo.PageParam;
import com.pei.dehaze.framework.common.validation.InEnum;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import org.springframework.format.annotation.DateTimeFormat;

import java.time.LocalDateTime;

import static com.pei.dehaze.framework.common.util.date.DateUtils.FORMAT_YEAR_MONTH_DAY_HOUR_MINUTE_SECOND;

@Schema(description = "管理后台 - AI 工作流分页 Request VO")
@Data
public class AiWorkflowPageReqVO extends PageParam {

    @Schema(description = "名称", example = "工作流")
    private String name;

    @Schema(description = "标识", example = "FLOW")
    private String code;

    @Schema(description = "状态", example = "1")
    @InEnum(CommonStatusEnum.class)
    private Integer status;

    @Schema(description = "创建时间")
    @DateTimeFormat(pattern = FORMAT_YEAR_MONTH_DAY_HOUR_MINUTE_SECOND)
    private LocalDateTime[] createTime;

}
