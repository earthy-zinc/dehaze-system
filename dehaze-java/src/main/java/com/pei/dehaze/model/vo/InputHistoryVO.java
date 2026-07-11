package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;

/**
 * 历史记录视图对象
 *
 * @author earthyzinc
 * @since 2024-06-12
 */
@Schema(description = "历史记录视图对象")
@Data
public class InputHistoryVO {

    @Schema(description = "记录ID")
    private Long id;

    @Schema(description = "原始图片URL")
    private String originalImageUrl;

    @Schema(description = "原始缩略图URL")
    private String originalThumbnailUrl;

    @Schema(description = "处理结果图片URL")
    private String resultImageUrl;

    @Schema(description = "结果缩略图URL")
    private String resultThumbnailUrl;

    @Schema(description = "算法ID")
    private Long algorithmId;

    @Schema(description = "算法名称")
    private String algorithmName;

    @Schema(description = "算法参数")
    private String algorithmParams;

    @Schema(description = "处理耗时（毫秒）")
    private Integer processingTime;

    @Schema(description = "处理状态")
    private Integer status;

    @Schema(description = "图片来源")
    private String inputSource;

    @Schema(description = "是否收藏")
    private Boolean isFavorite;

    @Schema(description = "同步状态")
    private Integer syncStatus;

    @Schema(description = "创建时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;
}
