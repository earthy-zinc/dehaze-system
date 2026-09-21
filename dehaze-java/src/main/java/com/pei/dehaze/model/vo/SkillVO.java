package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** Skill 详情（管理员全部字段，含指令全文与资源文件清单） */
@Schema(description = "Skill 详情视图")
@Data
public class SkillVO {

    private Long id;

    private String name;

    private String description;

    private String scene;

    @Schema(description = "SKILL.md 指令正文")
    private String instruction;

    private String license;

    private String compatibility;

    private Map<String, Object> metadata;

    private String allowedTools;

    private List<FileItem> files = new ArrayList<>();

    private List<SkippedFile> skippedFiles = new ArrayList<>();

    private Integer status;

    @Schema(description = "来源(builtin/admin)")
    private String source;

    @Schema(description = "被 Agent 关联数")
    private Integer agentCount = 0;

    @Schema(description = "是否共享至 Skill 市场(0:否;1:是)")
    private Integer marketShared = 0;

    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;

    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime updateTime;

    /** SKILL 目录内资源文件清单项（内容存对象存储，仅返回清单供按需加载） */
    @Schema(description = "SKILL 资源文件清单项")
    @Data
    public static class FileItem {

        private String path;

        private Long fileSize;

        private String fileType;
    }

    /** 上传时被跳过的资源文件（部分失败可见，避免静默丢弃） */
    @Schema(description = "SKILL 被跳过的资源文件")
    @Data
    public static class SkippedFile {

        private String path;

        private String reason;
    }
}
