package com.pei.dehaze.model.query;

import com.pei.dehaze.common.base.BasePageQuery;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * 外部 A2A 端点分页查询参数
 *
 * @author dehaze
 */
@Data
@EqualsAndHashCode(callSuper = true)
@Schema(description = "外部 A2A 端点分页查询参数")
public class AiEndpointPageQuery extends BasePageQuery {

    @Schema(description = "关键字(按名称/地址模糊搜索)")
    private String keyword;

    @Min(value = 0, message = "状态不能小于0")
    @Max(value = 1, message = "状态不能大于1")
    @Schema(description = "状态过滤")
    private Integer status;
}
