package com.pei.dehaze.sdk.model.auth;

import lombok.Data;

@Data
public class LoginResponse {
    private String sessionId;
    private LoginUser user;

    @Data
    public static class LoginUser {
        private long id;
        private String username;
        private String nickname;
    }
}
