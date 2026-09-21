package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.dehaze.model.entity.SysAiModel;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

@Mapper
public interface SysAiModelMapper extends BaseMapper<SysAiModel> {

    /**
     * chat 模型近 24h 调用统计（含失败/超时，成功率真实反映调用质量）。
     *
     * <p>返回行字段：model / total / ok / last_at。
     */
    @Select("""
            <script>
            SELECT model AS model, COUNT(*) AS total,
                   SUM(CASE WHEN status = 1 THEN 1 ELSE 0 END) AS ok,
                   MAX(create_time) AS last_at
            FROM sys_ai_llm_call
            WHERE create_time <![CDATA[>=]]> #{since} AND model IN
            <foreach collection="models" item="m" open="(" separator="," close=")">#{m}</foreach>
            GROUP BY model
            </script>
            """)
    List<Map<String, Object>> selectChatUsage24h(@Param("since") LocalDateTime since,
                                                 @Param("models") List<String> models);

    /**
     * embedding/rerank 模型近 24h 计费流水统计（成功调用才落账，恒为可用佐证）。
     */
    @Select("""
            <script>
            SELECT model AS model, COUNT(*) AS total, MAX(create_time) AS lastAt
            FROM sys_ai_billing
            WHERE create_time <![CDATA[>=]]> #{since}
              AND bill_type IN ('embedding', 'rerank') AND model IN
            <foreach collection="models" item="m" open="(" separator="," close=")">#{m}</foreach>
            GROUP BY model
            </script>
            """)
    List<Map<String, Object>> selectKbUsage24h(@Param("since") LocalDateTime since,
                                               @Param("models") List<String> models);

    /**
     * 正在使用该模型的活跃会话数。
     *
     * <p>直接读 sys_ai_conversation（会话实体归属 AI 对话模块，此处仅取计数避免跨模块耦合）。
     */
    @Select("SELECT COUNT(*) FROM sys_ai_conversation "
            + "WHERE model = #{modelId} AND deleted = 0 AND status = 1")
    long countActiveConversations(@Param("modelId") String modelId);

    /** 使用该模型的所有活跃会话用户 ID（去重，供模型下线通知） */
    @Select("SELECT DISTINCT user_id FROM sys_ai_conversation "
            + "WHERE model = #{modelId} AND deleted = 0 AND status = 1")
    List<Long> selectActiveConversationUserIds(@Param("modelId") String modelId);
}
