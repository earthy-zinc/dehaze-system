package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableLogic;
import com.pei.dehaze.common.base.BaseEntity;
import lombok.Data;
import lombok.EqualsAndHashCode;

/** 召回测试集（表 sys_knowledge_test_set）：一条问题 + 期望命中分块 */
@Data
@EqualsAndHashCode(callSuper = false)
public class SysKnowledgeTestSet extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long knowledgeBaseId;

    private String question;

    /** 期望命中分块 ID 数组（JSON 文本） */
    private String expectedChunkIds;

    @TableLogic(value = "0", delval = "id")
    private Long deleted;
}
