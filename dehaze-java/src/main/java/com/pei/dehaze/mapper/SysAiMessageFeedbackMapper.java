package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.extension.handlers.JacksonTypeHandler;
import com.pei.dehaze.model.entity.SysAiMessageFeedback;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Result;
import org.apache.ibatis.annotations.Results;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;

/**
 * AI 消息反馈访问层
 *
 * @author dehaze
 */
@Mapper
public interface SysAiMessageFeedbackMapper extends BaseMapper<SysAiMessageFeedback> {

    /**
     * 按 (message_id, user_id) 查询（含软删行：唯一键含 deleted，复活原行而非新增）
     */
    @Select("SELECT * FROM sys_ai_message_feedback WHERE message_id = #{messageId} AND user_id = #{userId} LIMIT 1")
    @Results({
            // 手写 SELECT 不经过 MP 自动 resultMap，tags(JSON 列) 必须显式指定 typeHandler，
            // 否则实体 tags 恒为 null，反馈详情读回标签为空数组
            @Result(column = "tags", property = "tags", typeHandler = JacksonTypeHandler.class)
    })
    SysAiMessageFeedback selectByUserAndMessageIgnoringDeleted(@Param("messageId") Long messageId,
                                                               @Param("userId") Long userId);

    /**
     * 复活并覆盖软删反馈行（deleted 归零）
     */
    @Update("UPDATE sys_ai_message_feedback SET rating = #{rating}, tags = #{tags}, `comment` = #{comment}, " +
            "conversation_id = #{conversationId}, model = #{model}, `source` = #{source}, processed = 0, " +
            "deleted = 0 WHERE id = #{id}")
    int revive(@Param("id") Long id, @Param("rating") Integer rating, @Param("tags") String tags,
               @Param("comment") String comment, @Param("conversationId") Long conversationId,
               @Param("model") String model, @Param("source") String source);

    /**
     * 撤销反馈：软删（deleted=1，与 python 口径一致）
     */
    @Update("UPDATE sys_ai_message_feedback SET deleted = 1 " +
            "WHERE message_id = #{messageId} AND user_id = #{userId} AND deleted = 0")
    int softDeleteByUserAndMessage(@Param("messageId") Long messageId, @Param("userId") Long userId);
}
