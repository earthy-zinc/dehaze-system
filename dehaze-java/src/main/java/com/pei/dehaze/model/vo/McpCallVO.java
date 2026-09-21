package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;

@Schema(description = "外部 MCP 工具调用审计记录")
@Data
public class McpCallVO {

    private Long id;

    private Long userId;

    private Long serverId;

    private String serverName;

    private String toolName;

    private String result;

    private Integer latencyMs;

    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;
}
