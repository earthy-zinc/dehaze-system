package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;

@Schema(description = "外部 MCP Server 视图")
@Data
public class McpServerVO {

    private Long id;

    private String name;

    private String description;

    private String protocolType;

    private String endpoint;

    private String authType;

    private Integer status;

    @Schema(description = "健康状态(online/offline)")
    private String health;

    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime lastCheckTime;

    private Integer toolCount;

    @Schema(description = "是否已配置凭据（密文本身不回显）")
    private Boolean credentialConfigured;

    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;

    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime updateTime;
}
