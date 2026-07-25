package com.pei.dehaze.model.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;

/**
 * 刷新令牌请求
 */
@Data
@Schema(description = "刷新令牌请求")
public class RefreshTokenForm {

    @Schema(description = "刷新令牌（使用 httpOnly Cookie 时可为空，从 Cookie 中读取）")
    private String refreshToken;
}
