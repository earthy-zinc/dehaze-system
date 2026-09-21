package com.pei.dehaze.model.form;

import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.databind.annotation.JsonNaming;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.Data;

/**
 * 复制智能体请求
 *
 * @author dehaze
 */
@Data
@JsonNaming(PropertyNamingStrategies.SnakeCaseStrategy.class)
public class AiAgentCopyForm {

    @NotBlank(message = "Agent编码不能为空")
    @Size(max = 64, message = "Agent编码长度不能超过64")
    @Schema(description = "新Agent唯一编码")
    private String agentCode;

}
