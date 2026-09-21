package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.Map;

/**
 * AI 中间产物响应
 *
 * @author dehaze
 */
@Data
public class AiArtifactVO {

    private Long id;

    private Long conversationId;

    private Long messageId;

    private String type;

    private String refType;

    private Long refId;

    private Map<String, Object> summary;

    private Integer isInvalid;

    @JsonInclude(JsonInclude.Include.NON_NULL)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;
}
