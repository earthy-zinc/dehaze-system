package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import lombok.Data;

import java.util.List;
import java.util.Map;

@Data
@Schema(description = "内部消息发送表单")
public class MessageSendForm {

    @Schema(description = "模板编码（使用模板时传入）")
    private String templateCode;

    @Schema(description = "消息类型", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotBlank(message = "消息类型不能为空")
    private String type;

    @Schema(description = "消息标题（未使用模板时必填）")
    private String title;

    @Schema(description = "消息正文（未使用模板时必填）")
    private String content;

    @Schema(description = "接收人ID列表", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotEmpty(message = "接收人列表不能为空")
    private List<Long> recipientIds;

    @Schema(description = "业务模块")
    private String bizModule;

    @Schema(description = "业务ID（用于幂等去重）")
    private String bizId;

    @Schema(description = "优先级")
    private Integer priority;

    @Schema(description = "跳转链接")
    private String jumpUrl;

    @Schema(description = "模板变量")
    private Map<String, String> variables;

    @Schema(description = "扩展数据")
    private Map<String, Object> extra;
}
