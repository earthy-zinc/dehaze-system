package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.Size;
import java.util.Map;
import lombok.Data;

/**
 * 更新定时任务请求
 *
 * <p>enabled 为 python {@code ScheduleUpdate.enabled} 的 {@code int|None = Field(None, ge=0, le=1)}，
 * 故只加范围、不加 {@code @NotNull}（缺省表示不更新该字段）。
 *
 * @author dehaze
 */
@Data
public class AiScheduleUpdateForm {

    @Size(max = 128, message = "任务名称长度不能超过128")
    @Schema(description = "任务名称")
    private String name;

    @Size(max = 64, message = "触发规则长度不能超过64")
    @Schema(description = "Cron触发规则")
    private String cron;

    @Size(max = 64, message = "时区长度不能超过64")
    @Schema(description = "任务时区")
    private String timezone;

    @Schema(description = "输入来源JSON")
    private Map<String, Object> input;

    @Schema(description = "输出目标JSON")
    private Map<String, Object> output;

    @Min(value = 0, message = "启停状态不能小于0")
    @Max(value = 1, message = "启停状态不能大于1")
    @Schema(description = "用户启停(1:启用;0:停用)")
    private Integer enabled;

}
