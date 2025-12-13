package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;

/**
 * 下载任务VO
 *
 * @author earthy-zinc
 * @since 2025-12-07
 */
@Data
@Schema(description = "下载任务信息")
public class DownloadTaskVO {

    @Schema(description = "任务ID")
    private String taskId;

    @Schema(description = "任务状态")
    private String status;

    @Schema(description = "进度（百分比）")
    private Integer progress;

    @Schema(description = "消息")
    private String message;

    @Schema(description = "下载链接")
    private String downloadUrl;

    @Schema(description = "过期时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime expireTime;

    @Schema(description = "创建时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;
}
