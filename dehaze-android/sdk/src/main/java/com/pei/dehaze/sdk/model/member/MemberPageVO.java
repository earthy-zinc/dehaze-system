package com.pei.dehaze.sdk.model.member;

import lombok.Data;

/**
 * 会员分页VO
 */
@Data
public class MemberPageVO {
    private Long userId;
    private String username;
    private String nickname;
    private String levelCode;
    private String levelName;
    private Integer growthValue;
    private Integer monthlyUsed;
    private String expireTime;
    private Integer status;
    private String becomeMemberTime;
}
