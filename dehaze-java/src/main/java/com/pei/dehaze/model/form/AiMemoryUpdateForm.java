package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.Size;
import lombok.Data;

/**
 * 更新记忆请求
 *
 * <p>status 为 python {@code MemoryUpdate.status} 的 {@code int|None = Field(None, ge=0, le=1)}，
 * 故只加范围、不加 {@code @NotNull}。
 *
 * @author dehaze
 */
@Data
public class AiMemoryUpdateForm {

    @Size(max = 2000, message = "记忆内容长度不能超过2000")
    @Schema(description = "记忆内容")
    private String content;

    @Min(value = 0, message = "重要性评分不能小于0")
    @Max(value = 100, message = "重要性评分不能大于100")
    @Schema(description = "重要性评分(0-100)")
    private Integer importance;

    @Min(value = 0, message = "状态不能小于0")
    @Max(value = 1, message = "状态不能大于1")
    @Schema(description = "状态(1:启用;0:禁用)")
    private Integer status;

}
