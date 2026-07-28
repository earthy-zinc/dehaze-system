package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonInclude;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@JsonInclude(JsonInclude.Include.NON_NULL)
@Schema(description = "会员分页列表视图对象")
public class MemberPageVO {

    @Schema(description = "用户ID")
    private Long userId;

    @Schema(description = "用户名")
    private String username;

    @Schema(description = "昵称")
    private String nickname;

    @Schema(description = "会员等级")
    private String levelCode;

    @Schema(description = "等级名称")
    private String levelName;

    @Schema(description = "成长值")
    private Long growthValue;

    @Schema(description = "本月已用次数（去雾+评估）")
    private Integer monthlyUsed;

    @Schema(description = "套餐到期时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime expireTime;

    @Schema(description = "会员状态(1:正常;0:冻结)")
    private Integer status;

    @Schema(description = "首次成为会员时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime becomeMemberTime;
}
