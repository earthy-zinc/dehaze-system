package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import java.util.List;
import java.util.Map;
import lombok.Data;

/**
 * 更新评测样本请求
 *
 * @author dehaze
 */
@Data
public class AiEvalSampleUpdateForm {

    @Schema(description = "任务目标")
    private String taskGoal;

    @Schema(description = "允许输入")
    private String allowedInput;

    @Schema(description = "期望工具调用序列(有序,每项 {\"name\",\"args\"},args 为期望参数子集)")
    private List<Map<String, Object>> tools;

    @Schema(description = "期望过程")
    private String expectedProcess;

    @Schema(description = "期望结果")
    private String expectedResult;

    @Schema(description = "禁止行为")
    private String forbiddenBehavior;

    @Schema(description = "风险等级(low/medium/high)")
    private String riskLevel;
}
