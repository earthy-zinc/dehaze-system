package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

@Data
@EqualsAndHashCode(callSuper = true)
@Schema(description = "消息模板详情视图对象")
public class MessageTemplateDetailVO extends MessageTemplateVO {

    @Schema(description = "正文模板")
    private String contentTemplate;

    @Schema(description = "默认推送渠道")
    private Map<String, Boolean> channels;

    @Schema(description = "变量定义")
    private List<Map<String, String>> variables;

    @Schema(description = "更新时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime updateTime;
}
