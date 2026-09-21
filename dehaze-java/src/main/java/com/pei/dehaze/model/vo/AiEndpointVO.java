package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.Map;

/**
 * 外部 A2A 端点响应（凭证不回显）
 *
 * @author dehaze
 */
@Data
public class AiEndpointVO {

    private Long id;

    private String name;

    private String agentCardUrl;

    private String baseUrl;

    private String authType;

    private Map<String, Object> agentCard;

    private Integer status;

    @JsonInclude(JsonInclude.Include.NON_NULL)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;
}
