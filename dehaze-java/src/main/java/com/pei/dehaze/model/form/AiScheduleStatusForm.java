package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

/**
 * 启停定时任务请求
 *
 * @author dehaze
 */
@Data
public class AiScheduleStatusForm {

    @NotNull(message = "启停状态不能为空")
    @Min(value = 0, message = "启停状态不能小于0")
    @Max(value = 1, message = "启停状态不能大于1")
    @Schema(description = "目标启停状态(1:启用;0:停用)")
    private Integer enabled;

}
