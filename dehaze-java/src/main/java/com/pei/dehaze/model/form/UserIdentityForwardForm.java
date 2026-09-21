package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

/**
 * 用户身份透传配置（抽象覆盖 DeepSeek user_id / OpenAI user / Anthropic metadata.user_id）。
 */
@Schema(description = "用户身份透传配置")
@Data
public class UserIdentityForwardForm {

    @Schema(description = "是否启用透传")
    private Boolean enabled;

    @Schema(description = "透传字段名或嵌套路径(user_id/user/metadata.user_id)")
    private String field;

    @Schema(description = "透传值脱敏前缀")
    private String prefix;

    @Schema(description = "透传值最大长度")
    private Integer maxLen;
}
