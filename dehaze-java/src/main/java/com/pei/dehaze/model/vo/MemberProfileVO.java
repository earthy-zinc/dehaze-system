package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonInclude;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@JsonInclude(JsonInclude.Include.NON_NULL)
@Schema(description = "会员信息视图对象（用户端profile）")
public class MemberProfileVO {

    @Schema(description = "用户ID")
    private Long userId;

    @Schema(description = "用户名")
    private String username;

    @Schema(description = "昵称")
    private String nickname;

    @Schema(description = "头像")
    private String avatar;

    @Schema(description = "会员等级")
    private String levelCode;

    @Schema(description = "等级名称")
    private String levelName;

    @Schema(description = "成长值")
    private Long growthValue;

    @Schema(description = "距下一等级成长值")
    private Long nextLevelGrowth;

    @Schema(description = "升级进度百分比")
    private Integer progressPercent;

    @Schema(description = "套餐到期时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime expireTime;

    @Schema(description = "本月去雾配额")
    private Integer monthlyDehazeQuota;

    @Schema(description = "本月已用去雾次数")
    private Integer monthlyDehazeUsed;

    @Schema(description = "本月评估配额")
    private Integer monthlyEvaluateQuota;

    @Schema(description = "本月已用评估次数")
    private Integer monthlyEvaluateUsed;

    @Schema(description = "权益配置")
    private BenefitVO benefits;

    @Schema(description = "会员状态(1:正常;0:冻结)")
    private Integer status;
}
