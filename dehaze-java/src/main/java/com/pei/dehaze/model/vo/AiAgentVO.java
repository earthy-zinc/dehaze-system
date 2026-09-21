package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * AI 智能体响应（列表项与详情共用，详情字段按需填充）
 *
 * @author dehaze
 */
@Data
public class AiAgentVO {

    private Long id;

    private String agentCode;

    private String name;

    private String description;

    private String modelId;

    private String reasoningMode;

    private Integer isSubagent;

    private Integer isTeam;

    private Integer isExposed;

    private List<String> tags;

    private Integer status;

    private Integer sortOrder;

    private Integer skillCount;

    private Integer mcpCount;

    private Integer subAgentCount;

    private String systemPrompt;

    private Map<String, Object> config;

    private List<Map<String, Object>> permissions;

    private List<String> skills;

    private List<String> mcpNamespaces;

    private List<AiSubAgentVO> subagents;

    @JsonInclude(JsonInclude.Include.NON_NULL)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;
}
