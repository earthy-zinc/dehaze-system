package com.pei.dehaze.module.ai.controller.admin.model.vo.apikey;

import lombok.*;
import io.swagger.v3.oas.annotations.media.Schema;
import com.pei.dehaze.framework.common.pojo.PageParam;

@Schema(description = "管理后台 - AI API 密钥分页 Request VO")
@Data
public class AiApiKeyPageReqVO extends PageParam {

    @Schema(description = "名称", example = "文心一言")
    private String name;

    @Schema(description = "平台", example = "OpenAI")
    private String platform;

    @Schema(description = "状态", example = "1")
    private Integer status;

}
