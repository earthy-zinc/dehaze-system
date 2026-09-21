package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import java.util.Map;
import jakarta.validation.constraints.Size;
import lombok.Data;

/**
 * 创建定时任务请求
 *
 * @author dehaze
 */
@Data
public class AiScheduleCreateForm {

    @NotBlank(message = "任务名称不能为空")
    @Size(max = 128, message = "任务名称长度不能超过128")
    @Schema(description = "任务名称")
    private String name;

    @NotBlank(message = "触发规则不能为空")
    @Size(max = 64, message = "触发规则长度不能超过64")
    @Schema(description = "Cron触发规则(5位表达式或常用频率标识)")
    private String cron;

    @Schema(description = "任务时区(默认Asia/Shanghai)")
    private String timezone;

    @Schema(description = "输入来源JSON")
    private Map<String, Object> input;

    @Schema(description = "输出目标JSON")
    private Map<String, Object> output;

}
