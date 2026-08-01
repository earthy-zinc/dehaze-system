package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@Schema(description = "收藏记录视图对象")
public class FavoriteVO {

    @Schema(description = "收藏记录ID")
    private Long id;

    @Schema(description = "用户ID")
    private Long userId;

    @Schema(description = "收藏对象类型")
    private String targetType;

    @Schema(description = "收藏对象ID")
    private Long targetId;

    @Schema(description = "收藏对象名称（关联查询）")
    private String targetName;

    @Schema(description = "对象摘要")
    private String targetSummary;

    @Schema(description = "缩略图URL")
    private String targetThumbnail;

    @Schema(description = "是否已失效（对象被删除）")
    private Boolean isInvalid;

    @Schema(description = "收藏时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;
}
