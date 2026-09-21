package com.pei.dehaze.model.query;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Min;
import lombok.Data;

/** 用户余额查询参数（对齐 python {@code /ai-billing/balance} 的 {@code userId} ge=1） */
@Schema(description = "用户余额查询参数")
@Data
public class AiBalanceQuery {

    /** 目标用户 ID（管理端下钻查询，需 ai:billing:stat；不传为当前登录用户） */
    @Min(1)
    private Long userId;
}
