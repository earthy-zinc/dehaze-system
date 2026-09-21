package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.pei.dehaze.model.entity.SysAiConversation;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;

import java.time.LocalDateTime;
import java.util.List;

/**
 * AI 会话访问层
 *
 * @author dehaze
 */
@Mapper
public interface SysAiConversationMapper extends BaseMapper<SysAiConversation> {

    /**
     * 回收站分页：已软删且未超 30 天恢复窗口，按 delete_time 倒序（绕过逻辑删除过滤）
     */
    @Select("SELECT * FROM sys_ai_conversation WHERE user_id = #{userId} AND deleted != 0 " +
            "AND delete_time >= #{windowStart} ORDER BY delete_time DESC, id DESC")
    IPage<SysAiConversation> selectTrashPage(IPage<SysAiConversation> page, @Param("userId") Long userId,
                                             @Param("windowStart") LocalDateTime windowStart);

    /**
     * 回收站单条（供恢复）：已软删且未超恢复窗口
     */
    @Select("SELECT * FROM sys_ai_conversation WHERE id = #{convId} AND user_id = #{userId} " +
            "AND deleted != 0 AND delete_time >= #{windowStart}")
    SysAiConversation selectInTrash(@Param("convId") Long convId, @Param("userId") Long userId,
                                    @Param("windowStart") LocalDateTime windowStart);

    /**
     * 软删并记录软删时间（delete_time 支撑 30 天恢复窗口）
     */
    @Update("<script>UPDATE sys_ai_conversation SET deleted = id, delete_time = NOW() " +
            "WHERE deleted = 0 AND id IN <foreach collection='ids' item='id' open='(' separator=',' close=')'>#{id}</foreach></script>")
    int softDeleteByIds(@Param("ids") List<Long> ids);

    /**
     * 恢复软删会话（清 deleted 与 delete_time）
     */
    @Update("<script>UPDATE sys_ai_conversation SET deleted = 0, delete_time = NULL " +
            "WHERE id IN <foreach collection='ids' item='id' open='(' separator=',' close=')'>#{id}</foreach></script>")
    int restoreByIds(@Param("ids") List<Long> ids);

    /**
     * 统计当前用户置顶且未删除的会话数（置顶上限校验）
     */
    @Select("SELECT COUNT(*) FROM sys_ai_conversation WHERE user_id = #{userId} AND deleted = 0 AND pinned = 1")
    long countActivePinned(@Param("userId") Long userId);

    @Update("UPDATE sys_ai_conversation SET pinned = #{pinned}, pinned_at = #{pinnedAt} WHERE id = #{convId}")
    int setPinned(@Param("convId") Long convId, @Param("pinned") Integer pinned,
                  @Param("pinnedAt") LocalDateTime pinnedAt);

    @Update("UPDATE sys_ai_conversation SET last_read_message_id = #{messageId} WHERE id = #{convId}")
    int markRead(@Param("convId") Long convId, @Param("messageId") Long messageId);

    @Update("<script>UPDATE sys_ai_conversation SET status = #{status} " +
            "WHERE id IN <foreach collection='ids' item='id' open='(' separator=',' close=')'>#{id}</foreach></script>")
    int updateStatusByIds(@Param("ids") List<Long> ids, @Param("status") Integer status);

    @Update("UPDATE sys_ai_conversation SET current_branch_message_id = #{messageId} WHERE id = #{convId}")
    int updateCurrentBranch(@Param("convId") Long convId, @Param("messageId") Long messageId);

    /**
     * 存在该 Agent 编码的会话数（删除 Agent 前的引用校验）
     */
    @Select("SELECT COUNT(*) FROM sys_ai_conversation WHERE agent_code = #{agentCode} AND deleted = 0")
    long countByAgentCode(@Param("agentCode") String agentCode);
}
