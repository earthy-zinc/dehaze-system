package com.pei.clarity.model.query;

import com.pei.clarity.common.base.BasePageQuery;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Schema(description = "知识库文档分页查询参数")
@Data
@EqualsAndHashCode(callSuper = true)
public class KbDocumentPageQuery extends BasePageQuery {

    @Schema(description = "处理状态过滤(pending/processing/completed/failed)")
    private String processingStatus;
}
