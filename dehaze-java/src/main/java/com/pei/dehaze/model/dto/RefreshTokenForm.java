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

    @NotBlank(message = "刷新令牌不能为空")
    @Schema(description = "刷新令牌")
    private String refreshToken;
}
