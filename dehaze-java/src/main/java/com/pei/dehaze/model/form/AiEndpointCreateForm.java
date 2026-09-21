package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.Data;

/**
 * 注册外部 A2A 端点请求
 *
 * @author dehaze
 */
@Data
public class AiEndpointCreateForm {

    @NotBlank(message = "端点名称不能为空")
    @Size(max = 128, message = "端点名称长度不能超过128")
    @Schema(description = "端点名称")
    private String name;

    @Size(max = 512, message = "Agent Card地址长度不能超过512")
    @Schema(description = "Agent Card地址")
    private String agentCardUrl;

    @NotBlank(message = "端点地址不能为空")
    @Size(max = 512, message = "端点地址长度不能超过512")
    @Schema(description = "A2A端点地址")
    private String baseUrl;

    @Schema(description = "认证方式(apiKey/http/oauth2/openIdConnect/mutualTLS)")
    private String authType;

    @Schema(description = "凭证(服务端AES加密存储)")
    private String credential;

    @Min(value = 0, message = "状态不能小于0")
    @Max(value = 1, message = "状态不能大于1")
    @Schema(description = "状态(1:启用;0:禁用)")
    private Integer status;

}
