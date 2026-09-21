package com.pei.dehaze.model.query;

import com.pei.dehaze.common.base.BasePageQuery;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Schema(description = "AI 供应商分页查询参数")
@Data
@EqualsAndHashCode(callSuper = true)
public class ProviderPageQuery extends BasePageQuery {

    @Schema(description = "关键字(按显示名称/供应商编码模糊搜索)")
    private String keyword;
}
