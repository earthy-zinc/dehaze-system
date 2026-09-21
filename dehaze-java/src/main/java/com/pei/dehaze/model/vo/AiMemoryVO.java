package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.Map;

/**
 * AI 长期记忆响应
 *
 * @author dehaze
 */
@Data
public class AiMemoryVO {

    private Long id;

    private Long userId;

    private String memoryType;

    private String content;

    private Map<String, Object> metadata;

    private Integer importance;

    private Integer accessCount;

    private LocalDateTime lastAccessedAt;

    private String source;

    private Integer status;

    private Integer archived;

    @JsonInclude(JsonInclude.Include.NON_NULL)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;

    @JsonInclude(JsonInclude.Include.NON_NULL)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime updateTime;
}
