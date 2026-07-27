package com.pei.dehaze.model.query;

import com.pei.dehaze.common.base.BasePageQuery;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@EqualsAndHashCode(callSuper = true)
@Schema(description = "消息搜索查询对象")
public class MessageSearchQuery extends BasePageQuery {

    @Schema(description = "搜索关键字", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotBlank(message = "搜索关键字不能为空")
    private String keyword;
}
