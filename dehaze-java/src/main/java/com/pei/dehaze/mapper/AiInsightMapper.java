package com.pei.dehaze.mapper;

import com.pei.dehaze.model.read.AgentRefCountRead;
import com.pei.dehaze.model.read.AnomalyStatusRead;
import com.pei.dehaze.model.read.ConversationConsumptionRead;
import com.pei.dehaze.model.read.DictItemRead;
import com.pei.dehaze.model.read.DisplayNameRead;
import com.pei.dehaze.model.read.DowngradeRead;
import com.pei.dehaze.model.read.ModelUsageRead;
import com.pei.dehaze.model.read.ProviderRead;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.time.LocalDateTime;
import java.util.List;

/**
 * AI 域跨模块只读聚合访问层（计费/过程链/Skill/MCP/用户），不承担写入。
 *
 * @author dehaze
 */
@Mapper
public interface AiInsightMapper {

    /**
     * 按会话聚合计费消耗：{conversationId, token(input+output), credits}
     */
    @Select("<script>SELECT conversation_id AS conversationId, " +
            "COALESCE(SUM(input_tokens + output_tokens), 0) AS token, " +
            "COALESCE(SUM(credits), 0) AS credits FROM sys_ai_billing " +
            "WHERE conversation_id IN <foreach collection='convIds' item='id' open='(' separator=',' close=')'>#{id}</foreach> " +
            "GROUP BY conversation_id</script>")
    List<ConversationConsumptionRead> sumConsumptionByConversationIds(@Param("convIds") List<Long> convIds);

    /**
     * 存在"连续配额不足"异常的会话 ID
     */
    @Select("<script>SELECT DISTINCT b.conversation_id FROM sys_ai_billing b " +
            "JOIN sys_ai_billing_anomaly a ON a.billing_id = b.id " +
            "WHERE a.anomaly_type = 'consecutive_quota_fail' AND b.conversation_id IN " +
            "<foreach collection='convIds' item='id' open='(' separator=',' close=')'>#{id}</foreach></script>")
    List<Long> listQuotaAnomalyConversationIds(@Param("convIds") List<Long> convIds);

    /**
     * 存在"工具调用失败"的会话 ID（高风险工具调用异常标注数据源）
     */
    @Select("<script>SELECT DISTINCT t.conversation_id FROM sys_ai_trace t " +
            "JOIN sys_ai_llm_call c ON c.trace_id = t.trace_id " +
            "WHERE t.conversation_id IN <foreach collection='convIds' item='id' open='(' separator=',' close=')'>#{id}</foreach> " +
            "AND c.tool_call IS NOT NULL AND c.status != 1</script>")
    List<Long> listRiskyToolConversationIds(@Param("convIds") List<Long> convIds);

    /**
     * 会话消息状态分布（异常标注：失败 3 / 取消 4）
     */
    @Select("<script>SELECT DISTINCT conversation_id AS conversationId, status FROM sys_ai_message " +
            "WHERE deleted = 0 AND status IN (3, 4) AND conversation_id IN " +
            "<foreach collection='convIds' item='id' open='(' separator=',' close=')'>#{id}</foreach></script>")
    List<AnomalyStatusRead> listAnomalyStatusByConversations(@Param("convIds") List<Long> convIds);

    /**
     * 用户展示名（昵称优先，回退用户名；含软删用户，审计需追溯历史归属）
     */
    @Select("<script>SELECT id, COALESCE(NULLIF(nickname, ''), username) AS name FROM sys_user WHERE id IN " +
            "<foreach collection='ids' item='id' open='(' separator=',' close=')'>#{id}</foreach></script>")
    List<DisplayNameRead> listUserDisplayNames(@Param("ids") List<Long> ids);

    @Select("<script>SELECT agent_id AS agentId, COUNT(*) AS cnt FROM sys_ai_agent_skill WHERE agent_id IN " +
            "<foreach collection='ids' item='id' open='(' separator=',' close=')'>#{id}</foreach> GROUP BY agent_id</script>")
    List<AgentRefCountRead> countSkillsByAgentIds(@Param("ids") List<Long> ids);

    @Select("<script>SELECT agent_id AS agentId, COUNT(*) AS cnt FROM sys_ai_agent_mcp WHERE agent_id IN " +
            "<foreach collection='ids' item='id' open='(' separator=',' close=')'>#{id}</foreach> GROUP BY agent_id</script>")
    List<AgentRefCountRead> countMcpByAgentIds(@Param("ids") List<Long> ids);

    @Select("<script>SELECT parent_agent_id AS agentId, COUNT(*) AS cnt FROM sys_ai_agent_subagent " +
            "WHERE parent_agent_id IN <foreach collection='ids' item='id' open='(' separator=',' close=')'>#{id}</foreach> " +
            "GROUP BY parent_agent_id</script>")
    List<AgentRefCountRead> countSubagentsByAgentIds(@Param("ids") List<Long> ids);

    /**
     * 引用计数：该 Agent 作为子 Agent 被引用的次数
     */
    @Select("SELECT COUNT(*) FROM sys_ai_agent_subagent WHERE subagent_agent_id = #{agentId}")
    long countSubagentReferences(@Param("agentId") Long agentId);

    /**
     * 启用中的供应商（运营统计看板）
     */
    @Select("SELECT id, display_name AS displayName FROM sys_ai_provider WHERE deleted = 0")
    List<ProviderRead> listProviders();

    @Select("<script>SELECT model AS modelId, COUNT(*) AS callCount, " +
            "COALESCE(SUM(input_tokens), 0) AS inputTokens, COALESCE(SUM(output_tokens), 0) AS outputTokens, " +
            "COALESCE(SUM(credits), 0) AS credits FROM sys_ai_billing WHERE 1 = 1 " +
            "<if test='startTime != null'> AND create_time &gt;= #{startTime}</if>" +
            "<if test='endTime != null'> AND create_time &lt;= #{endTime}</if> GROUP BY model</script>")
    List<ModelUsageRead> listModelUsage(@Param("startTime") LocalDateTime startTime,
                                        @Param("endTime") LocalDateTime endTime);

    @Select("<script>SELECT actual_model AS modelId, COUNT(*) AS cnt FROM sys_ai_billing " +
            "WHERE actual_model IS NOT NULL " +
            "<if test='startTime != null'> AND create_time &gt;= #{startTime}</if>" +
            "<if test='endTime != null'> AND create_time &lt;= #{endTime}</if> GROUP BY actual_model</script>")
    List<DowngradeRead> listDowngradeByModel(@Param("startTime") LocalDateTime startTime,
                                             @Param("endTime") LocalDateTime endTime);

    /**
     * 会员等级编码（定时调度等按等级开放的功能判定，无会员记录返回 null）
     */
    @Select("SELECT level_code FROM sys_member WHERE user_id = #{userId} AND deleted = 0 LIMIT 1")
    String selectMemberLevelCode(@Param("userId") Long userId);

    /**
     * 单个字典值（评测阈值等配置项，缺省回落代码常量）
     */
    @Select("SELECT value FROM sys_dict WHERE type_code = #{typeCode} AND name = #{name} " +
            "AND status = 1 AND deleted = 0 LIMIT 1")
    String selectDictValue(@Param("typeCode") String typeCode, @Param("name") String name);

    /**
     * 启用中的字典项（Agent 配置三级合并的护栏默认值来源）
     */
    @Select("SELECT name, value FROM sys_dict WHERE type_code = #{typeCode} AND status = 1 AND deleted = 0")
    List<DictItemRead> listEnabledDictItems(@Param("typeCode") String typeCode);

    @Select("<script>SELECT model_id AS modelId, display_name AS name FROM sys_ai_model WHERE model_id IN " +
            "<foreach collection='modelIds' item='id' open='(' separator=',' close=')'>#{id}</foreach></script>")
    List<java.util.Map<String, Object>> listModelDisplayNames(@Param("modelIds") List<String> modelIds);
}
