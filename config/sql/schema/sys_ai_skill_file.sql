-- ============================================================
-- 表名: sys_ai_skill_file
-- 模块: 核心模块-AI对话
-- ============================================================
-- 设计思路:
-- SKILL 目录文件清单表（Agent Skills 规范）：SKILL 是一个目录，除 SKILL.md
-- 正文（存 sys_ai_skill.instruction）外，可选 reference/ script/ assets/
-- README.md 等文件。文件内容存入对象存储（MinIO，对象 key = skills/{name}/{path}），
-- 本表仅存文件清单（path/大小/类型），支持渐进披露第三级（资源按需加载）
-- 与列表/详情展示，避免大文件内容撑爆 DB 与 SkillManager 内存缓存。
-- path 为相对 SKILL 根目录的路径（如 "reference/REFERENCE.md"、"script/run.py"）。
-- 删除/更新 Skill 时级联删除对象存储文件（Service 层处理，无物理外键，对齐项目惯例）。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_ai_skill_file`;
CREATE TABLE `sys_ai_skill_file`
(
    `id`          bigint                                                          NOT NULL AUTO_INCREMENT COMMENT '主键',
    `skill_id`    bigint                                                          NOT NULL COMMENT '所属Skill主键(sys_ai_skill.id)',
    `path`        varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci   NOT NULL COMMENT '相对SKILL根目录的文件路径',
    `file_size`   bigint                                                          NOT NULL DEFAULT 0 COMMENT '文件大小(字节)',
    `file_type`   varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci    NULL DEFAULT NULL COMMENT '文件类型(MIME或扩展名)',
    `create_by`   bigint                                                          NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`   bigint                                                          NULL DEFAULT NULL COMMENT '修改人ID',
    `create_time` datetime                                                        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time` datetime                                                        NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `uk_skill_path` (`skill_id`, `path`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = 'AI对话Skill目录文件清单表(F-M08-006)'
  ROW_FORMAT = DYNAMIC;
