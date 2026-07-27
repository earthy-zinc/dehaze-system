package com.pei.dehaze.model.query;

import com.pei.dehaze.common.base.BasePageQuery;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@EqualsAndHashCode(callSuper = true)
@Schema(description = "消息分页查询对象")
public class MessageQuery extends BasePageQuery {

    @Schema(description = "消息类型")
    private String type;

    @Schema(description = "已读状态(0:未读;1:已读)")
    private Integer readStatus;
}
