package com.pei.dehaze.model.query;

import com.pei.dehaze.common.base.BasePageQuery;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Schema(description = "Skill 列表查询参数")
@Data
@EqualsAndHashCode(callSuper = true)
public class SkillPageQuery extends BasePageQuery {

    @Schema(description = "关键字(按名称模糊搜索)")
    private String keyword;

    @Schema(description = "状态筛选(1:启用;0:禁用)")
    @Min(0)
    @Max(1)
    private Integer status;
}
