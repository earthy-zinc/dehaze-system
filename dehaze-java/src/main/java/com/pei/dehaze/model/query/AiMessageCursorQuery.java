package com.pei.dehaze.model.query;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import lombok.Data;

/**
 * 会话消息游标分页查询参数（id 倒序，契约三端一致）。
 *
 * <p>游标语义：仅返回 {@code id < before} 的消息；缺省 {@code before} 取最新一页。
 * 采用查询对象承载，约束以字段级注解声明、由控制层 {@code @Valid} 触发（与项目内分页
 * DTO 一致），而非方法级 {@code @Validated}——故本端点的越界拒绝发生在参数绑定阶段。
 *
 * @author dehaze
 */
@Data
@Schema(description = "会话消息游标分页查询参数")
public class AiMessageCursorQuery {

    @Schema(description = "游标：仅返回 id 小于该值的消息（缺省取最新一页）", example = "100")
    @Min(value = 1, message = "before 必须大于0")
    private Long before;

    @Schema(description = "每页条数（1..100）", defaultValue = "50", example = "50")
    @Min(value = 1, message = "每页大小必须大于0")
    @Max(value = 100, message = "每页大小不能超过100")
    private int limit = 50;
}
