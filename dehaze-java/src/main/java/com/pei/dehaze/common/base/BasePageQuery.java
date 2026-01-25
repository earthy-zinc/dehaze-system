package com.pei.dehaze.common.base;


import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import lombok.Data;

/**
 * 基础分页请求对象
 *
 * @author earthyzinc
 * @since 2021/2/28
 */
@Data
@Schema
public class BasePageQuery {

    @Schema(description = "页码", requiredMode = Schema.RequiredMode.REQUIRED, example = "1")
    @Min(value = 1, message = "页码必须大于0")
    private int pageNum = 1;

    @Schema(description = "每页记录数", requiredMode = Schema.RequiredMode.REQUIRED, example = "10")
    @Min(value = 1, message = "每页大小必须大于0")
    @Max(value = 100, message = "每页大小不能超过100")
    private int pageSize = 10;
}
