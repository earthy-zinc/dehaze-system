package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;

/**
 * 算法版本视图对象
 *
 * @author earthyzinc
 * @since 2024-06-12
 */
@Schema(description = "算法版本视图对象")
@Data
public class AlgorithmVersionVO {

    @Schema(description = "版本ID")
    private Long id;

    @Schema(description = "算法ID")
    private Long algorithmId;

    @Schema(description = "版本号")
    private String version;

    @Schema(description = "变更日志")
    private String changeLog;

    @Schema(description = "状态")
    private Integer status;

    @Schema(description = "是否当前活跃版本")
    private Boolean isActive;

    @Schema(description = "模型文件ID")
    private Long modelFileId;

    @Schema(description = "创建时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;
}
