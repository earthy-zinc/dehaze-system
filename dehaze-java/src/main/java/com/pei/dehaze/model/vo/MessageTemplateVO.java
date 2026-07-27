package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonInclude;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@JsonInclude(JsonInclude.Include.NON_NULL)
@Schema(description = "消息模板视图对象")
public class MessageTemplateVO {

    @Schema(description = "模板ID")
    private Long id;

    @Schema(description = "模板编码")
    private String code;

    @Schema(description = "模板名称")
    private String name;

    @Schema(description = "消息类型")
    private String type;

    @Schema(description = "标题模板")
    private String titleTemplate;

    @Schema(description = "默认优先级")
    private Integer priority;

    @Schema(description = "状态(1:启用;0:禁用)")
    private Integer status;

    @Schema(description = "创建时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;
}
