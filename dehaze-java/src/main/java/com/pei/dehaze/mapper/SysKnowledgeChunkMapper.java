package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.dehaze.model.entity.SysKnowledgeChunk;
import org.apache.ibatis.annotations.Delete;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

@Mapper
public interface SysKnowledgeChunkMapper extends BaseMapper<SysKnowledgeChunk> {

    /** 汇总文档下分块 token 总数（SQL 聚合，避免加载大字段） */
    @Select("SELECT COALESCE(SUM(token_count), 0) FROM sys_knowledge_chunk WHERE document_id = #{documentId}")
    long sumTokensByDocument(@Param("documentId") Long documentId);

    /** 清除文档全部分块（文档删除/版本更新时调用） */
    @Delete("DELETE FROM sys_knowledge_chunk WHERE document_id = #{documentId}")
    int deleteByDocument(@Param("documentId") Long documentId);
}
