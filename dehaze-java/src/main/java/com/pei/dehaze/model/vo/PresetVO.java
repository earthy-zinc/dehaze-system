package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.v3.oas.annotations.media.Schema;
import java.time.LocalDateTime;
import lombok.Data;

@Schema(description = "参数预设视图对象")
@Data
public class PresetVO {

    @Schema(description = "预设ID")
    private Long id;

    @Schema(description = "预设名称")
    private String name;

    @Schema(description = "预设类型(system:系统预设;custom:用户自定义)")
    private String type;

    @Schema(description = "关联算法ID")
    private String modelId;

    @Schema(description = "参数键值对(JSON)")
    private String params;

    @Schema(description = "所属用户ID(系统预设为空)")
    private Long userId;

    @Schema(description = "是否默认预设")
    private Integer isDefault;

    @Schema(description = "创建时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;
}
