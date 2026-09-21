package com.pei.dehaze.model.query;

import com.pei.dehaze.common.base.BasePageQuery;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Schema(description = "知识库分页查询参数")
@Data
@EqualsAndHashCode(callSuper = true)
public class KbPageQuery extends BasePageQuery {

    @Schema(description = "关键字(按知识库名称模糊搜索)")
    private String keyword;

    @Schema(description = "管理端视角(admin:返回全部知识库含私有库,仅管理员)")
    private String view;
}
