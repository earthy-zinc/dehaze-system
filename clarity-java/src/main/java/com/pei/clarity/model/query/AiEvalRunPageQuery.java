package com.pei.clarity.model.query;

import com.pei.clarity.common.base.BasePageQuery;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * 智能体评测执行记录分页查询参数。
 *
 * <p>对齐 clarity-python {@code list_runs} 的查询参数（pageNum/pageSize/datasetId）；
 * 分页边界由 {@link BasePageQuery} 的 {@code @Min/@Max} 约束（pageSize ≤ 100，越界 A0400）。
 *
 * @author clarity
 */
@Data
@EqualsAndHashCode(callSuper = true)
@Schema(description = "评测执行记录分页查询参数")
public class AiEvalRunPageQuery extends BasePageQuery {

    @Schema(description = "评测集ID过滤")
    private Long datasetId;
}
