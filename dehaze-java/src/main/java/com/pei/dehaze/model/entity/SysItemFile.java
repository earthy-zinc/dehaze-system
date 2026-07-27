package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.FieldFill;
import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableLogic;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;

@Data
public class SysItemFile {
    @TableId(type = IdType.AUTO)
    private Long id;

    private Long itemId;

    private Long fileId;

    private Long thumbnailFileId;

    private String type;

    private String description;

    @Schema(description = "场景类型")
    private String sceneType;

    @Schema(description = "雾霾程度")
    private String hazeLevel;

    @Schema(description = "图片宽度")
    private Integer width;

    @Schema(description = "图片高度")
    private Integer height;

    @Schema(description = "使用次数")
    private Long usageCount;

    /**
     * 逻辑删除标识(0:未删除;1:已删除)
     */
    @TableLogic
    private Integer deleted;

    @TableField(fill = FieldFill.INSERT)
    private LocalDateTime createTime;

    @TableField(fill = FieldFill.INSERT_UPDATE)
    private LocalDateTime updateTime;
}
