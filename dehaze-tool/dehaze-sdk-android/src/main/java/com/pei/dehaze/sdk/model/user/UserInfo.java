package com.pei.dehaze.sdk.model.user;

import lombok.Data;
import java.util.Date;
import java.util.List;

/**
 * 登录用户信息模型类
 */
@Data
public class UserInfo {
    private Integer userId;
    private String username;
    private String nickname;
    private String avatar;
    private List<String> roles;
    private List<String> perms;
}