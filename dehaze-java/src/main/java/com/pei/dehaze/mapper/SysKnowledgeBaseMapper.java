package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.dehaze.model.entity.SysKnowledgeBase;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Update;

@Mapper
public interface SysKnowledgeBaseMapper extends BaseMapper<SysKnowledgeBase> {

    /**
     * 原子累加知识库冗余统计（文档数/分块数/Token 总数）。
     *
     * <p>单条 UPDATE 内自增自减，避免 python 侧 CAS 重试循环的读改写窗口。
     */
    @Update("UPDATE sys_knowledge_base SET document_count = document_count + #{documentDelta}, "
            + "chunk_count = chunk_count + #{chunkDelta}, total_tokens = total_tokens + #{tokenDelta} "
            + "WHERE id = #{kbId} AND document_count + #{documentDelta} >= 0 "
            + "AND chunk_count + #{chunkDelta} >= 0 AND total_tokens + #{tokenDelta} >= 0")
    int updateStats(@Param("kbId") Long kbId,
                    @Param("documentDelta") int documentDelta,
                    @Param("chunkDelta") int chunkDelta,
                    @Param("tokenDelta") long tokenDelta);
}
