package com.pei.dehaze.model.form;

import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.databind.annotation.JsonNaming;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import java.util.List;
import lombok.Data;

/**
 * 创建评测样本请求
 *
 * @author dehaze
 */
@Data
@JsonNaming(PropertyNamingStrategies.SnakeCaseStrategy.class)
public class AiEvalSampleCreateForm {

    @NotNull(message = "评测集ID不能为空")
    @Schema(description = "关联评测集ID")
    private Long datasetId;

    @NotBlank(message = "任务目标不能为空")
    @Schema(description = "任务目标")
    private String taskGoal;

    @Schema(description = "允许输入")
    private String allowedInput;

    @Schema(description = "可用工具")
    private List<String> tools;

    @Schema(description = "期望过程")
    private String expectedProcess;

    @Schema(description = "期望结果")
    private String expectedResult;

    @Schema(description = "禁止行为")
    private String forbiddenBehavior;

    @Schema(description = "风险等级(low/medium/high)")
    private String riskLevel;

}
