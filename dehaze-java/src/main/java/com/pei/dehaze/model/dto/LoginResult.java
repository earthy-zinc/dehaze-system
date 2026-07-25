package com.pei.dehaze.model.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Builder;
import lombok.Data;

@Schema(description = "登录响应对象")
@Data
@Builder
public class LoginResult {

    @Schema(description = "会话ID（Web端自动通过Cookie传递，移动端需手动设置X-Session-Id请求头）")
    private String sessionId;

    @Schema(description = "用户信息")
    private UserInfo user;

    @Data
    @Builder
    public static class UserInfo {
        private Long id;
        private String username;
        private String nickname;
    }
}
