package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.Map;

/**
 * 智能体版本响应（snapshot 仅在版本详情端点返回）
 *
 * @author dehaze
 */
@Data
public class AiAgentVersionVO {

    private Long id;

    private Long agentId;

    private Integer versionNo;

    private Integer status;

    private String changeNote;

    private Long operatorId;

    private Map<String, Object> snapshot;

    @JsonInclude(JsonInclude.Include.NON_NULL)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;
}
