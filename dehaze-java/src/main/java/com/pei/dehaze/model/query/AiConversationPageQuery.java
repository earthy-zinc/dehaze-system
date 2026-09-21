package com.pei.dehaze.model.query;

import com.pei.dehaze.common.base.BasePageQuery;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * AI 会话分页查询参数
 *
 * @author dehaze
 */
@Data
@EqualsAndHashCode(callSuper = true)
@Schema(description = "AI 会话分页查询参数")
public class AiConversationPageQuery extends BasePageQuery {

    @Schema(description = "关键字(标题/消息内容)")
    private String keyword;

    @Schema(description = "会话状态范围过滤(0:全部;1:活跃,默认;2:已归档)")
    private Integer status;

    @Schema(description = "视角(admin:管理端会话审计,读取全量用户会话,需 ai:conversation:audit)")
    private String view;
}
