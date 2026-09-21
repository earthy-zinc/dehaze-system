package com.pei.dehaze.model.query;

import com.pei.dehaze.common.base.BasePageQuery;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * 长期记忆分页查询参数（活跃列表与归档列表共用，对齐 dehaze-python {@code MemoryPageQuery}）。
 *
 * <p>分页边界由 {@link BasePageQuery} 的 {@code @Min/@Max} 约束（pageSize ≤ 100，越界 A0400）。
 *
 * @author dehaze
 */
@Data
@EqualsAndHashCode(callSuper = true)
@Schema(description = "长期记忆分页查询参数")
public class AiMemoryPageQuery extends BasePageQuery {

    @Schema(description = "记忆类型过滤")
    private String memoryType;

    @Schema(description = "来源过滤(conversation/feedback/reflection/manual)")
    private String source;
}
