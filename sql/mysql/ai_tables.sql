CREATE DATABASE IF NOT EXISTS `pei_ai`;
USE `pei_ai`;

-- ----------------------------
-- Table structure for ai_chat_conversation
-- ----------------------------
DROP TABLE IF EXISTS `ai_chat_conversation`;
CREATE TABLE `ai_chat_conversation`
(
    `id`             bigint        NOT NULL AUTO_INCREMENT COMMENT '对话主键',
    `user_id`        bigint        NOT NULL DEFAULT 0 COMMENT '用户编号',
    `title`          varchar(255)  NOT NULL DEFAULT '' COMMENT '对话标题',
    `pinned`         bit(1)        NOT NULL DEFAULT b'0' COMMENT '是否置顶',
    `pinned_time`    datetime      NULL     DEFAULT NULL COMMENT '置顶时间',
    `role_id`        bigint        NOT NULL DEFAULT 0 COMMENT '角色编号',
    `model_id`       bigint        NOT NULL DEFAULT 0 COMMENT '模型编号',
    `model`          varchar(64)   NOT NULL DEFAULT '' COMMENT '模型标志',
    `system_message` varchar(1024) NULL     DEFAULT NULL COMMENT '角色设定',
    `temperature`    decimal(3, 2) NULL     DEFAULT NULL COMMENT '温度参数',
    `max_tokens`     int           NULL     DEFAULT NULL COMMENT '单条回复的最大Token数量',
    `max_contexts`   int           NULL     DEFAULT NULL COMMENT '上下文的最大Message数量',
    `creator`        varchar(64)   NULL     DEFAULT '' COMMENT '创建者',
    `create_time`    datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`        varchar(64)   NULL     DEFAULT '' COMMENT '更新者',
    `update_time`    datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`        bit(1)        NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_user_id` (`user_id` ASC) USING BTREE,
    INDEX `idx_role_id` (`role_id` ASC) USING BTREE,
    INDEX `idx_model_id` (`model_id` ASC) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'AI Chat 对话表';

-- ----------------------------
-- Table structure for ai_chat_message
-- ----------------------------
DROP TABLE IF EXISTS `ai_chat_message`;
CREATE TABLE `ai_chat_message`
(
    `id`              bigint      NOT NULL AUTO_INCREMENT COMMENT '消息主键',
    `conversation_id` bigint      NOT NULL DEFAULT 0 COMMENT '对话编号',
    `reply_id`        bigint      NULL     DEFAULT NULL COMMENT '回复消息编号',
    `type`            varchar(32) NOT NULL DEFAULT '' COMMENT '消息类型。user: 用户；assistant: 助手；system: 系统；tool: 工具',
    `user_id`         bigint      NOT NULL DEFAULT 0 COMMENT '用户编号',
    `role_id`         bigint      NOT NULL DEFAULT 0 COMMENT '角色编号',
    `model`           varchar(64) NOT NULL DEFAULT '' COMMENT '模型标志',
    `model_id`        bigint      NOT NULL DEFAULT 0 COMMENT '模型编号',
    `content`         text        NULL     DEFAULT NULL COMMENT '聊天内容',
    `use_context`     bit(1)      NOT NULL DEFAULT b'0' COMMENT '是否携带上下文',
    `segment_ids`     text        NULL     DEFAULT NULL COMMENT '知识库段落编号数组（逗号分隔）',
    `creator`         varchar(64) NULL     DEFAULT '' COMMENT '创建者',
    `create_time`     datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`         varchar(64) NULL     DEFAULT '' COMMENT '更新者',
    `update_time`     datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`         bit(1)      NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_conversation_id` (`conversation_id` ASC) USING BTREE,
    INDEX `idx_user_id` (`user_id` ASC) USING BTREE,
    INDEX `idx_role_id` (`role_id` ASC) USING BTREE,
    INDEX `idx_model_id` (`model_id` ASC) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'AI Chat 消息表';

-- ----------------------------
-- Table structure for ai_image
-- ----------------------------
DROP TABLE IF EXISTS `ai_image`;
CREATE TABLE `ai_image`
(
    `id`            bigint       NOT NULL AUTO_INCREMENT COMMENT '图片主键',
    `user_id`       bigint       NOT NULL DEFAULT 0 COMMENT '用户编号',
    `prompt`        text         NULL COMMENT '提示词',
    `platform`      varchar(32)  NOT NULL DEFAULT '' COMMENT '平台',
    `model_id`      bigint       NOT NULL DEFAULT 0 COMMENT '模型编号',
    `model`         varchar(64)  NOT NULL DEFAULT '' COMMENT '模型标识',
    `width`         int          NULL     DEFAULT NULL COMMENT '图片宽度',
    `height`        int          NULL     DEFAULT NULL COMMENT '图片高度',
    `status`        int          NOT NULL DEFAULT 0 COMMENT '生成状态',
    `finish_time`   datetime     NULL     DEFAULT NULL COMMENT '完成时间',
    `error_message` text         NULL COMMENT '绘画错误信息',
    `pic_url`       varchar(512) NOT NULL DEFAULT '' COMMENT '图片地址',
    `public_status` bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否公开',
    `options`       text         NULL COMMENT '绘制参数',
    `buttons`       text         NULL COMMENT 'MJ按钮',
    `task_id`       varchar(255) NOT NULL DEFAULT '' COMMENT '任务编号',
    `creator`       varchar(64)  NULL     DEFAULT '' COMMENT '创建者',
    `create_time`   datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`       varchar(64)  NULL     DEFAULT '' COMMENT '更新者',
    `update_time`   datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`       bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_user_id` (`user_id` ASC) USING BTREE,
    INDEX `idx_model_id` (`model_id` ASC) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'AI 绘画表';

-- ----------------------------
-- Table structure for ai_knowledge
-- ----------------------------
DROP TABLE IF EXISTS `ai_knowledge`;
CREATE TABLE `ai_knowledge`
(
    `id`                   bigint        NOT NULL AUTO_INCREMENT COMMENT '知识库主键',
    `name`                 varchar(255)  NOT NULL DEFAULT '' COMMENT '知识库名称',
    `description`          text          NULL COMMENT '知识库描述',
    `embedding_model_id`   bigint        NOT NULL DEFAULT 0 COMMENT '向量模型编号',
    `embedding_model`      varchar(64)   NOT NULL DEFAULT '' COMMENT '模型标识',
    `top_k`                int           NULL     DEFAULT NULL COMMENT 'TopK值',
    `similarity_threshold` decimal(5, 2) NULL     DEFAULT NULL COMMENT '相似度阈值',
    `status`               int           NOT NULL DEFAULT 0 COMMENT '状态',
    `creator`              varchar(64)   NULL     DEFAULT '' COMMENT '创建者',
    `create_time`          datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`              varchar(64)   NULL     DEFAULT '' COMMENT '更新者',
    `update_time`          datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`              bit(1)        NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_embedding_model_id` (`embedding_model_id` ASC) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'AI 知识库表';

-- ----------------------------
-- Table structure for ai_knowledge_document
-- ----------------------------
DROP TABLE IF EXISTS `ai_knowledge_document`;
CREATE TABLE `ai_knowledge_document`
(
    `id`                 bigint       NOT NULL AUTO_INCREMENT COMMENT '文档主键',
    `knowledge_id`       bigint       NOT NULL DEFAULT 0 COMMENT '知识库编号',
    `name`               varchar(255) NOT NULL DEFAULT '' COMMENT '文档名称',
    `url`                varchar(512) NOT NULL DEFAULT '' COMMENT '文件URL',
    `content`            longtext     NULL COMMENT '内容',
    `content_length`     int          NULL     DEFAULT NULL COMMENT '文档长度',
    `tokens`             int          NULL     DEFAULT NULL COMMENT '文档token数量',
    `segment_max_tokens` int          NULL     DEFAULT NULL COMMENT '分片最大Token数',
    `retrieval_count`    int          NOT NULL DEFAULT 0 COMMENT '召回次数',
    `status`             int          NOT NULL DEFAULT 0 COMMENT '状态',
    `creator`            varchar(64)  NULL     DEFAULT '' COMMENT '创建者',
    `create_time`        datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`            varchar(64)  NULL     DEFAULT '' COMMENT '更新者',
    `update_time`        datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`            bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_knowledge_id` (`knowledge_id` ASC) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'AI 知识库文档表';

-- ----------------------------
-- Table structure for ai_knowledge_segment
-- ----------------------------
DROP TABLE IF EXISTS `ai_knowledge_segment`;
CREATE TABLE `ai_knowledge_segment`
(
    `id`              bigint       NOT NULL AUTO_INCREMENT COMMENT '分段主键',
    `knowledge_id`    bigint       NOT NULL DEFAULT 0 COMMENT '知识库编号',
    `document_id`     bigint       NOT NULL DEFAULT 0 COMMENT '文档编号',
    `content`         longtext     NULL COMMENT '切片内容',
    `content_length`  int          NULL     DEFAULT NULL COMMENT '切片内容长度',
    `vector_id`       varchar(255) NOT NULL DEFAULT '' COMMENT '向量库编号',
    `tokens`          int          NULL     DEFAULT NULL COMMENT 'token数量',
    `retrieval_count` int          NOT NULL DEFAULT 0 COMMENT '召回次数',
    `status`          int          NOT NULL DEFAULT 0 COMMENT '状态',
    `creator`         varchar(64)  NULL     DEFAULT '' COMMENT '创建者',
    `create_time`     datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`         varchar(64)  NULL     DEFAULT '' COMMENT '更新者',
    `update_time`     datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`         bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_knowledge_id` (`knowledge_id` ASC) USING BTREE,
    INDEX `idx_document_id` (`document_id` ASC) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'AI 知识库文档分段表';

-- ----------------------------
-- Table structure for ai_mind_map
-- ----------------------------
DROP TABLE IF EXISTS `ai_mind_map`;
CREATE TABLE `ai_mind_map`
(
    `id`                bigint      NOT NULL AUTO_INCREMENT COMMENT '思维导图主键',
    `user_id`           bigint      NOT NULL DEFAULT 0 COMMENT '用户编号',
    `platform`          varchar(32) NOT NULL DEFAULT '' COMMENT '平台',
    `model_id`          bigint      NOT NULL DEFAULT 0 COMMENT '模型编号',
    `model`             varchar(64) NOT NULL DEFAULT '' COMMENT '模型',
    `prompt`            text        NULL COMMENT '生成内容提示',
    `generated_content` text        NULL COMMENT '生成的内容',
    `error_message`     text        NULL COMMENT '错误信息',
    `creator`           varchar(64) NULL     DEFAULT '' COMMENT '创建者',
    `create_time`       datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`           varchar(64) NULL     DEFAULT '' COMMENT '更新者',
    `update_time`       datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`           bit(1)      NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_user_id` (`user_id` ASC) USING BTREE,
    INDEX `idx_model_id` (`model_id` ASC) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'AI 思维导图表';

-- ----------------------------
-- Table structure for ai_api_key
-- ----------------------------
DROP TABLE IF EXISTS `ai_api_key`;
CREATE TABLE `ai_api_key`
(
    `id`          bigint       NOT NULL AUTO_INCREMENT COMMENT '秘钥主键',
    `name`        varchar(255) NOT NULL DEFAULT '' COMMENT '名称',
    `api_key`     varchar(512) NOT NULL DEFAULT '' COMMENT 'API密钥',
    `platform`    varchar(32)  NOT NULL DEFAULT '' COMMENT '平台',
    `url`         varchar(512) NOT NULL DEFAULT '' COMMENT 'API地址',
    `status`      int          NOT NULL DEFAULT 0 COMMENT '状态',
    `creator`     varchar(64)  NULL     DEFAULT '' COMMENT '创建者',
    `create_time` datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`     varchar(64)  NULL     DEFAULT '' COMMENT '更新者',
    `update_time` datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`     bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'AI API秘钥表';

-- ----------------------------
-- Table structure for ai_chat_role
-- ----------------------------
DROP TABLE IF EXISTS `ai_chat_role`;
CREATE TABLE `ai_chat_role`
(
    `id`             bigint       NOT NULL AUTO_INCREMENT COMMENT '角色主键',
    `name`           varchar(255) NOT NULL DEFAULT '' COMMENT '角色名称',
    `avatar`         varchar(512) NOT NULL DEFAULT '' COMMENT '角色头像',
    `category`       varchar(255) NOT NULL DEFAULT '' COMMENT '角色分类',
    `description`    text         NULL COMMENT '角色描述',
    `system_message` text         NULL COMMENT '角色设定',
    `user_id`        bigint       NOT NULL DEFAULT 0 COMMENT '用户编号',
    `model_id`       bigint       NOT NULL DEFAULT 0 COMMENT '模型编号',
    `knowledge_ids`  text         NULL     DEFAULT NULL COMMENT '引用的知识库编号列表（逗号分隔）',
    `tool_ids`       text         NULL     DEFAULT NULL COMMENT '引用的工具编号列表（逗号分隔）',
    `public_status`  bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否公开',
    `sort`           int          NOT NULL DEFAULT 0 COMMENT '排序值',
    `status`         int          NOT NULL DEFAULT 0 COMMENT '状态。0: 启用；1: 停用',
    `creator`      varchar(64)   NULL     DEFAULT '' COMMENT '创建者',
    `create_time`  datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`      varchar(64)   NULL     DEFAULT '' COMMENT '更新者',
    `update_time`  datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`      bit(1)        NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_user_id` (`user_id` ASC) USING BTREE,
    INDEX `idx_model_id` (`model_id` ASC) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'AI 聊天角色表';

-- ----------------------------
-- Table structure for ai_model
-- ----------------------------
DROP TABLE IF EXISTS `ai_model`;
CREATE TABLE `ai_model`
(
    `id`           bigint        NOT NULL AUTO_INCREMENT COMMENT '模型主键',
    `key_id`       bigint        NOT NULL DEFAULT 0 COMMENT 'API秘钥编号',
    `name`         varchar(255)  NOT NULL DEFAULT '' COMMENT '模型名称',
    `model`        varchar(64)   NOT NULL DEFAULT '' COMMENT '模型标志',
    `platform`     varchar(32)   NOT NULL DEFAULT '' COMMENT '平台',
    `type`         int           NOT NULL DEFAULT 0 COMMENT '类型',
    `sort`         int           NOT NULL DEFAULT 0 COMMENT '排序值',
    `status`       int           NOT NULL DEFAULT 0 COMMENT '状态',
    `temperature`  decimal(3, 2) NULL     DEFAULT NULL COMMENT '温度参数',
    `max_tokens`   int           NULL     DEFAULT NULL COMMENT '单条回复的最大Token数量',
    `max_contexts` int           NULL     DEFAULT NULL COMMENT '上下文的最大Message数量',
    `creator`      varchar(64)   NULL     DEFAULT '' COMMENT '创建者',
    `create_time`  datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`      varchar(64)   NULL     DEFAULT '' COMMENT '更新者',
    `update_time`  datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`      bit(1)        NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_key_id` (`key_id` ASC) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'AI 模型表';

-- ----------------------------
-- Table structure for ai_tool
-- ----------------------------
DROP TABLE IF EXISTS `ai_tool`;
CREATE TABLE `ai_tool`
(
    `id`          bigint       NOT NULL AUTO_INCREMENT COMMENT '工具主键',
    `name`        varchar(255) NOT NULL DEFAULT '' COMMENT '工具名称',
    `description` text         NULL COMMENT '工具描述',
    `status`      int          NOT NULL DEFAULT 0 COMMENT '状态',
    `creator`     varchar(64)  NULL     DEFAULT '' COMMENT '创建者',
    `create_time` datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`     varchar(64)  NULL     DEFAULT '' COMMENT '更新者',
    `update_time` datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`     bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'AI 工具表';

-- ----------------------------
-- Table structure for ai_music
-- ----------------------------
DROP TABLE IF EXISTS `ai_music`;
CREATE TABLE `ai_music`
(
    `id`            bigint         NOT NULL AUTO_INCREMENT COMMENT '音乐主键',
    `user_id`       bigint         NOT NULL DEFAULT 0 COMMENT '用户编号',
    `title`         varchar(255)   NOT NULL DEFAULT '' COMMENT '音乐名称',
    `lyric`         varchar(2048)  NULL     DEFAULT NULL COMMENT '歌词',
    `image_url`     varchar(512)   NOT NULL DEFAULT '' COMMENT '图片地址',
    `audio_url`     varchar(512)   NOT NULL DEFAULT '' COMMENT '音频地址',
    `video_url`     varchar(512)   NOT NULL DEFAULT '' COMMENT '视频地址',
    `status`        int            NOT NULL DEFAULT 0 COMMENT '音乐状态。10: 进行中；20: 已完成；30: 已失败',
    `generate_mode` int            NOT NULL DEFAULT 0 COMMENT '生成模式。1: 描述模式；2: 歌词模式',
    `description`   varchar(1024)  NULL     DEFAULT NULL COMMENT '描述词',
    `platform`      varchar(32)    NOT NULL DEFAULT '' COMMENT '平台。TongYi: 通义千问；YiYan: 文心一言；DeepSeek: DeepSeek；ZhiPu: 智谱；XingHuo: 星火；DouBao: 豆包；HunYuan: 混元；SiliconFlow: 硅基流动；MiniMax: MiniMax；Moonshot: 月之暗灭；BaiChuan: 百川智能；OpenAI: OpenAI；AzureOpenAI: AzureOpenAI；Ollama: Ollama；StableDiffusion: StableDiffusion；Midjourney: Midjourney；Suno: Suno',
    `model`         varchar(64)    NOT NULL DEFAULT '' COMMENT '模型',
    `tags`          text           NULL     DEFAULT NULL COMMENT '音乐风格标签（JSON格式）',
    `duration`      decimal(10, 2) NULL     DEFAULT NULL COMMENT '音乐时长',
    `public_status` bit(1)         NOT NULL DEFAULT b'0' COMMENT '是否公开',
    `task_id`       varchar(255)   NOT NULL DEFAULT '' COMMENT '任务编号',
    `error_message` varchar(1024)  NULL     DEFAULT NULL COMMENT '错误信息',
    `creator`       varchar(64)    NULL     DEFAULT '' COMMENT '创建者',
    `create_time`   datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`       varchar(64)    NULL     DEFAULT '' COMMENT '更新者',
    `update_time`   datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`       bit(1)         NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_user_id` (`user_id` ASC) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'AI 音乐表';

-- ----------------------------
-- Table structure for ai_workflow
-- ----------------------------
DROP TABLE IF EXISTS `ai_workflow`;
CREATE TABLE `ai_workflow`
(
    `id`          bigint       NOT NULL AUTO_INCREMENT COMMENT '工作流主键',
    `name`        varchar(255) NOT NULL DEFAULT '' COMMENT '工作流名称',
    `code`        varchar(255) NOT NULL DEFAULT '' COMMENT '工作流标识',
    `graph`       longtext     NULL COMMENT '工作流模型JSON数据',
    `remark`      text         NULL COMMENT '备注',
    `status`      int          NOT NULL DEFAULT 0 COMMENT '状态',
    `creator`     varchar(64)  NULL     DEFAULT '' COMMENT '创建者',
    `create_time` datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`     varchar(64)  NULL     DEFAULT '' COMMENT '更新者',
    `update_time` datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`     bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `idx_code` (`code` ASC) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'AI 工作流表';

-- ----------------------------
-- Table structure for ai_write
-- ----------------------------
DROP TABLE IF EXISTS `ai_write`;
CREATE TABLE `ai_write`
(
    `id`                bigint        NOT NULL AUTO_INCREMENT COMMENT '写作主键',
    `user_id`           bigint        NOT NULL DEFAULT 0 COMMENT '用户编号',
    `type`              int           NOT NULL DEFAULT 0 COMMENT '写作类型。1: 撰写；2: 回复',
    `platform`          varchar(32)   NOT NULL DEFAULT '' COMMENT '平台。TongYi: 通义千问；YiYan: 文心一言；DeepSeek；ZhiPu: 智谱；XingHuo: 星火；DouBao: 豆包；HunYuan: 混元；SiliconFlow: 硅基流动；MiniMax；Moonshot: 月之暗灭；BaiChuan: 百川智能；OpenAI；AzureOpenAI；Ollama；StableDiffusion；Midjourney；Suno',
    `model_id`          bigint        NOT NULL DEFAULT 0 COMMENT '模型编号',
    `model`             varchar(64)   NOT NULL DEFAULT '' COMMENT '模型',
    `prompt`            varchar(2048) NULL     DEFAULT NULL COMMENT '生成内容提示',
    `generated_content` text          NULL     DEFAULT NULL COMMENT '生成的内容',
    `original_content`  text          NULL     DEFAULT NULL COMMENT '原文',
    `length`            int           NULL     DEFAULT NULL COMMENT '长度提示词。参考 ai_write_length 字典',
    `format`            int           NULL     DEFAULT NULL COMMENT '格式提示词。参考 ai_write_format 字典',
    `tone`              int           NULL     DEFAULT NULL COMMENT '语气提示词。参考 ai_write_tone 字典',
    `language`          int           NULL     DEFAULT NULL COMMENT '语言提示词。参考 ai_write_language 字典',
    `error_message`     varchar(1024) NULL     DEFAULT NULL COMMENT '错误信息',
    `creator`           varchar(64)   NULL     DEFAULT '' COMMENT '创建者',
    `create_time`       datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`           varchar(64)   NULL     DEFAULT '' COMMENT '更新者',
    `update_time`       datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`           bit(1)        NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_user_id` (`user_id` ASC) USING BTREE,
    INDEX `idx_model_id` (`model_id` ASC) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'AI 写作表';
