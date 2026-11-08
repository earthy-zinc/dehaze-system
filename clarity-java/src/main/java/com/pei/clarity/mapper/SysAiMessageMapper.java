package com.pei.clarity.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.clarity.model.entity.SysAiMessage;
import java.util.List;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;

/**
 * AI 消息访问层
 *
 * @author clarity
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
    @Select("SELECT id FROM sys_ai_message WHERE conversation_id = #{convId} AND deleted = 0 "
            + "ORDER BY create_time DESC, id DESC LIMIT 1")
    Long getLastMessageId(@Param("convId") Long convId);

    /**
     * 消息按关键词定位：{conversationId, messageId}，取每个会话命中的最新一条（搜索命中正文时前端定位）
     *
     * <p>正文已并入 {@code blocks}（Block[] JSON），检索按 JSON 文本 LIKE 命中（无 ES 时的等价定位）。
     * {@code keyword} 由调用方先转义 {@code \ % _}；此处显式 {@code ESCAPE '\\'}，与 Python/Go 同口径。
     */
    @Select("<script>SELECT m.conversation_id AS conversationId, MAX(m.id) AS messageId FROM sys_ai_message m "
            + "WHERE m.deleted = 0 AND m.blocks LIKE CONCAT('%', #{keyword}, '%') ESCAPE '\\\\' "
            + "AND m.conversation_id IN <foreach collection='convIds' item='id' open='(' separator=',' close=')'>#{id}</foreach> "
            + "GROUP BY m.conversation_id</script>")
    List<KeywordMatchRow> findLatestIdsByKeyword(
            @Param("convIds") List<Long> convIds, @Param("keyword") String keyword);

    /**
     * 关键词命中消息所属会话 ID（无 ES 时以 DB LIKE 检索消息正文）
     *
     * <p>{@code keyword} 由调用方先转义 {@code \ % _}；此处显式 {@code ESCAPE '\\'}，与 Python/Go 同口径。
     */
    @Select("<script>SELECT DISTINCT conversation_id FROM sys_ai_message WHERE deleted = 0 "
            + "AND blocks LIKE CONCAT('%', #{keyword}, '%') ESCAPE '\\\\'</script>")
    List<Long> listConversationIdsByKeyword(@Param("keyword") String keyword);

    @Update("UPDATE sys_ai_message SET status = #{status} WHERE id = #{msgId}")
    int updateStatus(@Param("msgId") Long msgId, @Param("status") String status);

    @Update(
            "<script>UPDATE sys_ai_message SET deleted = id "
                    + "WHERE deleted = 0 AND id IN <foreach collection='ids' item='id' open='(' separator=',' close=')'>#{id}</foreach></script>")
    int softDeleteByIds(@Param("ids") List<Long> ids);

    /**
     * 会话最大未删除消息 ID（分支指针缺失/悬空时回退的链尾）
     */
    @Select("SELECT MAX(id) FROM sys_ai_message WHERE conversation_id = #{convId} AND deleted = 0")
    Long getMaxMessageId(@Param("convId") Long convId);

    /**
     * 批量取指定父消息下未删 assistant 兄弟：按 {@code parent_message_id ASC, id ASC} 返回。
     *
     * <p>一次 IN 查询覆盖整页消息的兄弟分组，避免逐条消息各查一次（N+1）。siblingIndex 即分组内
     * 下标 + 1，siblingCount 即分组长度；root 级（{@code parent_message_id IS NULL}）不参与。
     */
    @Select("<script>SELECT parent_message_id AS parentMessageId, id AS messageId FROM sys_ai_message "
            + "WHERE conversation_id = #{convId} AND deleted = 0 AND role = 'assistant' "
            + "AND parent_message_id IN <foreach collection='parentIds' item='id' open='(' separator=',' close=')'>#{id}</foreach> "
            + "ORDER BY parent_message_id ASC, id ASC</script>")
    List<SiblingRow> listAssistantSiblingIds(@Param("convId") Long convId, @Param("parentIds") List<Long> parentIds);

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

    /**
     * 兄弟分组行（自定义查询投影）
     */
    class SiblingRow {
        private Long parentMessageId;
        private Long messageId;

        public Long getParentMessageId() {
            return parentMessageId;
        }

        public void setParentMessageId(Long parentMessageId) {
            this.parentMessageId = parentMessageId;
        }

        public Long getMessageId() {
            return messageId;
        }

        public void setMessageId(Long messageId) {
            this.messageId = messageId;
        }
    }
}
