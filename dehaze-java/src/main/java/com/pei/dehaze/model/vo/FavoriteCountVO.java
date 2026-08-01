package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Schema(description = "收藏数量统计（按类型分组）")
public class FavoriteCountVO {

    @Schema(description = "收藏对象类型")
    private String targetType;

    @Schema(description = "该类型收藏数量")
    private Long count;
}
