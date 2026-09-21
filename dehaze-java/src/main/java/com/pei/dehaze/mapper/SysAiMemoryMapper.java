package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.dehaze.model.entity.SysAiMemory;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;

import java.util.List;

/**
 * AI 长期记忆访问层
 *
 * @author dehaze
 */
@Mapper
public interface SysAiMemoryMapper extends BaseMapper<SysAiMemory> {

    /**
     * 软删并记录软删时间（deleted=id + delete_time），供 30 天恢复窗口判定
     */
    @Update("<script>UPDATE sys_ai_memory SET deleted = id, delete_time = NOW(), update_by = #{operatorId} " +
            "WHERE deleted = 0 AND id IN <foreach collection='ids' item='id' open='(' separator=',' close=')'>#{id}</foreach></script>")
    int softDeleteByIds(@Param("ids") List<Long> ids, @Param("operatorId") Long operatorId);

    /**
     * 批量清空：全部 / 指定类型 / 指定时间范围（按 create_time），软删并记录 delete_time
     */
    @Update("<script>UPDATE sys_ai_memory SET deleted = id, delete_time = NOW(), update_by = #{operatorId} " +
            "WHERE user_id = #{userId} AND deleted = 0 " +
            "<if test='memoryType != null'> AND memory_type = #{memoryType}</if>" +
            "<if test='start != null'> AND create_time &gt;= #{start}</if>" +
            "<if test='end != null'> AND create_time &lt;= #{end}</if></script>")
    int batchClear(@Param("userId") Long userId, @Param("memoryType") String memoryType,
                   @Param("start") java.time.LocalDateTime start, @Param("end") java.time.LocalDateTime end,
                   @Param("operatorId") Long operatorId);

    /**
     * 恢复窗口内的软删记忆（绕过逻辑删除过滤）
     */
    @Select("<script>SELECT * FROM sys_ai_memory WHERE user_id = #{userId} AND deleted != 0 " +
            "AND delete_time &gt;= #{windowStart} " +
            "<if test='memoryType != null'> AND memory_type = #{memoryType}</if>" +
            "<if test='start != null'> AND create_time &gt;= #{start}</if>" +
            "<if test='end != null'> AND create_time &lt;= #{end}</if></script>")
    List<SysAiMemory> listDeletedForRestore(@Param("userId") Long userId, @Param("memoryType") String memoryType,
                                            @Param("start") java.time.LocalDateTime start,
                                            @Param("end") java.time.LocalDateTime end,
                                            @Param("windowStart") java.time.LocalDateTime windowStart);

    /**
     * 恢复软删记忆（清 deleted 与 delete_time）
     */
    @Update("<script>UPDATE sys_ai_memory SET deleted = 0, delete_time = NULL, update_by = #{operatorId} " +
            "WHERE id IN <foreach collection='ids' item='id' open='(' separator=',' close=')'>#{id}</foreach></script>")
    int restoreByIds(@Param("ids") List<Long> ids, @Param("operatorId") Long operatorId);

    /**
     * 检索命中后重激活：访问计数 +1、重置衰减计时器、重要性 +5（上限 100）
     */
    @Update("UPDATE sys_ai_memory SET access_count = access_count + 1, last_accessed_at = NOW(), " +
            "importance = LEAST(100, importance + 5) WHERE id = #{id}")
    int touch(@Param("id") Long id);
}
