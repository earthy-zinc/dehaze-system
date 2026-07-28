package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@EqualsAndHashCode(callSuper = true)
@Schema(description = "评价列表VO（后台）")
public class RatingPageVO extends MyRatingVO {

    @Schema(description = "用户ID")
    private Long userId;

    @Schema(description = "用户名")
    private String username;

    @Schema(description = "用户头像")
    private String userAvatar;

    @Schema(description = "是否隐藏(0:否;1:是)")
    private Integer isHidden;
}
