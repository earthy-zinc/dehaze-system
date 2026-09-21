package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

/**
 * 启停智能体请求
 *
 * @author dehaze
 */
@Data
public class AiAgentStatusForm {

    @NotNull(message = "状态不能为空")
    @Min(value = 0, message = "状态不能小于0")
    @Max(value = 1, message = "状态不能大于1")
    @Schema(description = "目标状态(1:启用;0:禁用)")
    private Integer status;

}
