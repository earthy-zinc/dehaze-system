package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.util.List;

/**
 * 会话消息游标分页响应（契约三端一致：{@code { list, total, hasMore }}）。
 *
 * <p>不走通用 {@code PageResult}：游标分页没有页码，且需要 {@code hasMore}
 * 表达"是否还存在比本页最后一条更早的消息"，通用分页结构无法承载。
 *
 * @author dehaze
 */
@Data
@Schema(description = "会话消息游标分页结果")
public class AiMessagePageVO {

    @Schema(description = "消息列表（id 倒序）")
    private List<AiMessageVO> list;

    @Schema(description = "该会话消息总数")
    private long total;

    @Schema(description = "是否还存在比本页最后一条更早的消息")
    private boolean hasMore;
}
