package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.dehaze.model.entity.SysAiMessage;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;

import java.util.List;

/**
 * AI 消息访问层
 *
 * @author dehaze
 */
@Mapper
public interface SysAiMessageMapper extends BaseMapper<SysAiMessage> {

    /**
     * 已读水位之后的消息数（会话未读数）
     */
    @Select("SELECT COUNT(*) FROM sys_ai_message WHERE conversation_id = #{convId} AND deleted = 0 AND id > #{afterId}")
    long countMessagesAfter(@Param("convId") Long convId, @Param("afterId") Long afterId);

    /**
     * 会话最后一条未删除消息 ID（标记已读用）
     */
    @Select("SELECT id FROM sys_ai_message WHERE conversation_id = #{convId} AND deleted = 0 " +
            "ORDER BY create_time DESC, id DESC LIMIT 1")
    Long getLastMessageId(@Param("convId") Long convId);

    /**
     * 消息按关键词定位：{conversationId, messageId}，取每个会话命中的最新一条（搜索命中内容时前端定位）
     */
    @Select("<script>SELECT m.conversation_id AS conversationId, MAX(m.id) AS messageId FROM sys_ai_message m " +
            "WHERE m.deleted = 0 AND m.content LIKE CONCAT('%', #{keyword}, '%') " +
            "AND m.conversation_id IN <foreach collection='convIds' item='id' open='(' separator=',' close=')'>#{id}</foreach> " +
            "GROUP BY m.conversation_id</script>")
    List<KeywordMatchRow> findLatestIdsByKeyword(@Param("convIds") List<Long> convIds, @Param("keyword") String keyword);

    /**
     * 关键词命中消息所属会话 ID（无 ES 时以 DB LIKE 检索消息正文）
     */
    @Select("<script>SELECT DISTINCT conversation_id FROM sys_ai_message WHERE deleted = 0 " +
            "AND content LIKE CONCAT('%', #{keyword}, '%')</script>")
    List<Long> listConversationIdsByKeyword(@Param("keyword") String keyword);

    @Update("UPDATE sys_ai_message SET status = #{status} WHERE id = #{msgId}")
    int updateStatus(@Param("msgId") Long msgId, @Param("status") Integer status);

    @Update("<script>UPDATE sys_ai_message SET deleted = id " +
            "WHERE deleted = 0 AND id IN <foreach collection='ids' item='id' open='(' separator=',' close=')'>#{id}</foreach></script>")
    int softDeleteByIds(@Param("ids") List<Long> ids);

    /**
     * 关键词命中行（自定义查询投影）
     */
    class KeywordMatchRow {
        private Long conversationId;
        private Long messageId;

        public Long getConversationId() {
            return conversationId;
        }

        public void setConversationId(Long conversationId) {
            this.conversationId = conversationId;
        }

        public Long getMessageId() {
            return messageId;
        }

        public void setMessageId(Long messageId) {
            this.messageId = messageId;
        }
    }
}
