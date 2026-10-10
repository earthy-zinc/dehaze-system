package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableLogic;
import com.pei.dehaze.common.base.BaseEntity;
import java.time.LocalDateTime;
import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * AI 模型配置（表 sys_ai_model）。
 *
 * <p>extra_request_params 为厂商私有 JSON 请求参数，实体按 TEXT 原样承载，
 * 序列化/反序列化在服务层完成，避免 JSON 列在两端（Java/Python）写法不一致。
 */
@Data
@EqualsAndHashCode(callSuper = false)
public class SysAiModel extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long providerId;

    /** 模型标识（业务键，如 gpt-4o），与 providerId 联合唯一 */
    private String modelId;

    /** 全模态模型类型（chat/embedding/rerank/image_gen/image_edit/video_gen/asr/tts/music_gen/3d_gen），创建后不可改 */
    private String modelType;

    /**
     * 任务归类标签（JSON 数组，按 TEXT 原样承载，服务层序列化/反序列化）。
     *
     * <p>图像模型 ["dehaze"]/["super_resolution"]，asr ["stream"]/["offline"]，embedding ["text"]/["image"]。
     */
    private String taskTags;

    /** embedding 向量维度，仅 embedding 类型有值，创建后不可改 */
    private Long dimension;

    private String displayName;

    private Integer maxContextTokens;

    private Integer maxOutputTokens;

    private Integer supportsMultimodal;

    private Integer supportsToolCall;

    private Integer supportsStreaming;

    private Integer supportsPromptCache;

    private Integer supportsStructuredOutput;

    private String extraRequestParams;

    /** 降级模型主键（sys_ai_model.id），为空表示无降级 */
    private Long fallbackModelId;

    private Integer promptCachePrefixLen;

    private Integer status;

    private Integer lastTestStatus;

    private LocalDateTime lastTestAt;

    private String lastTestError;

    /** 最低可用 VIP 等级（0:所有用户;1:VIP1及以上;2:VIP2及以上） */
    private Integer vipLevel;

    @TableLogic(value = "0", delval = "id")
    private Long deleted;
}
