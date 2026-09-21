package com.pei.dehaze.model.query;

import com.pei.dehaze.common.base.BasePageQuery;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Schema(description = "AI 模型分页查询参数")
@Data
@EqualsAndHashCode(callSuper = true)
public class AiModelPageQuery extends BasePageQuery {

    @Schema(description = "关键字(按显示名称/模型标识模糊搜索)")
    private String keyword;

    @Schema(description = "按模型类型筛选(chat/embedding/rerank)")
    private String modelType;
}
