package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.dehaze.model.entity.SysKnowledgeChunkFeedback;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;
import java.util.Map;

@Mapper
public interface SysKnowledgeChunkFeedbackMapper extends BaseMapper<SysKnowledgeChunkFeedback> {

    /** 知识库下被点踩的分块数（去重） */
    @Select("SELECT COUNT(DISTINCT c.id) FROM sys_knowledge_chunk_feedback fb "
            + "JOIN sys_knowledge_chunk c ON c.id = fb.chunk_id "
            + "WHERE c.knowledge_base_id = #{kbId} AND fb.rating = -1")
    long countLowQualityByKb(@Param("kbId") Long kbId);

    /** 知识库下被点踩的分块清单（按点踩次数降序），字段名与 python 仓储同口径 */
    @Select("SELECT c.id AS chunkId, c.content AS content, c.document_id AS documentId, "
            + "COUNT(fb.id) AS thumbsDownCount "
            + "FROM sys_knowledge_chunk_feedback fb "
            + "JOIN sys_knowledge_chunk c ON c.id = fb.chunk_id "
            + "WHERE c.knowledge_base_id = #{kbId} AND fb.rating = -1 "
            + "GROUP BY c.id, c.content, c.document_id "
            + "ORDER BY thumbsDownCount DESC, c.id ASC "
            + "LIMIT #{size} OFFSET #{offset}")
    List<Map<String, Object>> listLowQualityByKb(@Param("kbId") Long kbId,
                                                 @Param("size") int size,
                                                 @Param("offset") int offset);
}
