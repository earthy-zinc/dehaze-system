package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Schema(description = "收藏状态")
public class FavoriteStatusVO {

    @Schema(description = "收藏对象类型")
    private String targetType;

    @Schema(description = "收藏对象ID")
    private Long targetId;

    @Schema(description = "是否已收藏")
    private Boolean favorited;
}
