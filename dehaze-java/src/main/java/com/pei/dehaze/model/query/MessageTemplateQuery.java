package com.pei.dehaze.model.query;

import com.pei.dehaze.common.base.BasePageQuery;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@EqualsAndHashCode(callSuper = true)
@Schema(description = "消息模板分页查询对象")
public class MessageTemplateQuery extends BasePageQuery {

    @Schema(description = "模板名称（模糊）")
    private String name;

    @Schema(description = "消息类型")
    private String type;

    @Schema(description = "状态(1:启用;0:禁用)")
    private Integer status;
}
