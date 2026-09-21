package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.Map;

/**
 * 推理步骤响应
 *
 * @author dehaze
 */
@Data
public class AiAgentThoughtVO {

    private Long id;

    private Long messageId;

    private Long conversationId;

    private Integer position;

    private String agentCode;

    private Integer isSubagent;

    private String thought;

    private String tool;

    private Map<String, Object> toolInput;

    private String observation;

    private Integer status;

    private Integer latencyMs;

    private String error;

    @JsonInclude(JsonInclude.Include.NON_NULL)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;
}
