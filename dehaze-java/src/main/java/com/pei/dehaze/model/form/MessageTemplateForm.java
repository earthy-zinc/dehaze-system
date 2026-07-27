package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.util.Map;

@Data
@Schema(description = "消息模板表单")
public class MessageTemplateForm {

    @Schema(description = "模板名称")
    private String name;

    @Schema(description = "标题模板")
    private String titleTemplate;

    @Schema(description = "正文模板")
    private String contentTemplate;

    @Schema(description = "默认优先级")
    private Integer priority;

    @Schema(description = "默认推送渠道")
    private Map<String, Boolean> channels;

    @Schema(description = "状态(1:启用;0:禁用)")
    private Integer status;
}
