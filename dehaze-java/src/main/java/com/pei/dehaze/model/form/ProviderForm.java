package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.Data;

import java.util.Map;

@Schema(description = "AI 供应商创建表单")
@Data
public class ProviderForm {

    @NotBlank(message = "供应商编码不能为空")
    @Size(min = 1, max = 32)
    private String providerCode;

    @NotBlank(message = "显示名称不能为空")
    @Size(min = 1, max = 128)
    private String displayName;

    @NotBlank(message = "API 基础地址不能为空")
    @Size(min = 1, max = 512)
    private String apiBaseUrl;

    @Schema(description = "协议类型(openai_compat:OpenAI兼容;anthropic:Claude原生)")
    @Size(max = 32)
    private String protocolType = "openai_compat";

    @Schema(description = "认证方式(bearer/x-api-key/custom)")
    @Size(max = 32)
    private String authType = "bearer";

    private Map<String, Object> defaultHeaders;

    private Integer sortOrder = 0;

    private Integer healthCheckEnabled = 1;

    private UserIdentityForwardForm userIdentityForward;

    private String remark;

    private Integer status = 1;
}
