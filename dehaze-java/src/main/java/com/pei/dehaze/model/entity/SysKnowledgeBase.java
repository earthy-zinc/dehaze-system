package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableLogic;
import com.pei.dehaze.common.base.BaseEntity;
import lombok.Data;
import lombok.EqualsAndHashCode;

import java.math.BigDecimal;

/** AI 知识库主表（表 sys_knowledge_base） */
@Data
@EqualsAndHashCode(callSuper = false)
public class SysKnowledgeBase extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private String name;

    private String description;

    /** public:平台公共库全员只读; private:私有库仅创建者可读写 */
    private String visibility;

    private String embeddingProvider;

    private String embeddingModel;

    /** fixed/semantic/recursive/qa/table，创建后不可改 */
    private String chunkingStrategy;

    private Integer chunkSize;

    private Integer chunkOverlap;

    /** vector/keyword/hybrid */
    private String searchStrategy;

    private BigDecimal hybridWeight;

    private Integer topK;

    private BigDecimal scoreThreshold;

    private Integer enableRerank;

    private String rerankModel;

    private Integer documentCount;

    private Integer chunkCount;

    private Long totalTokens;

    /** 1:启用;2:处理中;0:禁用 */
    private Integer status;

    @TableLogic(value = "0", delval = "id")
    private Long deleted;
}
