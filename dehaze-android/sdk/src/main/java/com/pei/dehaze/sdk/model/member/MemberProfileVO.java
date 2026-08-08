package com.pei.dehaze.sdk.model.member;

import lombok.Data;

/**
 * 会员信息VO（用户端profile）
 */
@Data
public class MemberProfileVO {
    private Long userId;
    private String username;
    private String nickname;
    private String avatar;
    private String levelCode;
    private String levelName;
    private Integer growthValue;
    private Integer nextLevelGrowth;
    private Integer progressPercent;
    private String expireTime;
    private Integer monthlyDehazeQuota;
    private Integer monthlyDehazeUsed;
    private Integer monthlyEvaluateQuota;
    private Integer monthlyEvaluateUsed;
    private BenefitVO benefits;
    private Integer status;
}
