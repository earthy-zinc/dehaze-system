package com.pei.dehaze.mapper;

import com.pei.dehaze.model.entity.SysAiTrace;
import com.pei.dehaze.model.query.AiObservabilityTraceQuery;
import com.pei.dehaze.model.read.AiTraceCostRead;
import com.pei.dehaze.model.read.AiTraceCostTrendRead;
import com.pei.dehaze.model.read.AiTraceTrendRead;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.time.LocalDateTime;
import java.util.List;

/**
 * AI 可观测性只读聚合访问层（过程链检索/消耗聚合/性能趋势），不承担写入。
 *
 * <p>过程链检索的会话归属与标题关键词共用一次会话表关联，避免重复 join 产生笛卡尔放大；
 * 分页由显式 LIMIT/OFFSET 控制（检索含条件 join，交由 MyBatis-Plus 自动 count 易被 join 优化误判）。
 */
@Mapper
public interface AiObservabilityMapper {

    @Select("SELECT COUNT(*) FROM sys_ai_trace WHERE status = #{status}")
    long countByStatus(@Param("status") Integer status);

    /** 配额拒绝类过程链数（按采集链路写入的拒绝类 error_type 统计） */
    @Select("<script>SELECT COUNT(*) FROM sys_ai_trace WHERE error_type IN " +
            "<foreach collection='types' item='type' open='(' separator=',' close=')'>#{type}</foreach></script>")
    long countQuotaRejected(@Param("types") List<String> types);

    /**
     * 高风险调用数：推理步数超阈值，或存在"发起工具调用但调用失败/超时"的 LLM 调用。
     *
     * <p>工具执行异常经恢复中间件兜住转为 ToolMessage，不落到 error_type；
     * 采集侧工具痕迹在 sys_ai_llm_call.tool_call（仅发起工具调用的轮次非空）。
     */
    @Select("SELECT COUNT(*) FROM sys_ai_trace t WHERE t.step_count >= #{threshold} OR EXISTS (" +
            "SELECT 1 FROM sys_ai_llm_call c WHERE c.trace_id = t.trace_id " +
            "AND c.tool_call IS NOT NULL AND c.status != 1)")
    long countHighRisk(@Param("threshold") int threshold);

    @Select("<script>SELECT COUNT(*) FROM sys_ai_trace t " +
            "<if test='q.userId != null or q.keyword != null'>JOIN sys_ai_conversation c ON c.id = t.conversation_id </if>" +
            "WHERE 1 = 1 " +
            "<if test='q.conversationId != null'> AND t.conversation_id = #{q.conversationId}</if>" +
            "<if test='q.userId != null'> AND c.user_id = #{q.userId} AND c.deleted = 0</if>" +
            "<if test='q.status != null'> AND t.status = #{q.status}</if>" +
            "<if test='q.agentCode != null'> AND t.agent_code = #{q.agentCode}</if>" +
            "<if test='q.model != null'> AND t.model = #{q.model}</if>" +
            "<if test='q.errorType != null'> AND t.error_type = #{q.errorType}</if>" +
            "<if test='q.keyword != null'> AND (t.trace_id LIKE CONCAT('%', #{q.keyword}, '%') " +
            "OR c.title LIKE CONCAT('%', #{q.keyword}, '%'))</if>" +
            "<if test='q.capability != null'> AND JSON_SEARCH(t.context_snapshot, 'one', #{q.capability}, NULL, '$.items[*].type') IS NOT NULL</if>" +
            "<if test='start != null'> AND t.create_time &gt;= #{start}</if>" +
            "<if test='end != null'> AND t.create_time &lt;= #{end}</if>" +
            "</script>")
    long countTraces(@Param("q") AiObservabilityTraceQuery query,
                     @Param("start") LocalDateTime start,
                     @Param("end") LocalDateTime end);

    @Select("<script>SELECT t.* FROM sys_ai_trace t " +
            "<if test='q.userId != null or q.keyword != null'>JOIN sys_ai_conversation c ON c.id = t.conversation_id </if>" +
            "WHERE 1 = 1 " +
            "<if test='q.conversationId != null'> AND t.conversation_id = #{q.conversationId}</if>" +
            "<if test='q.userId != null'> AND c.user_id = #{q.userId} AND c.deleted = 0</if>" +
            "<if test='q.status != null'> AND t.status = #{q.status}</if>" +
            "<if test='q.agentCode != null'> AND t.agent_code = #{q.agentCode}</if>" +
            "<if test='q.model != null'> AND t.model = #{q.model}</if>" +
            "<if test='q.errorType != null'> AND t.error_type = #{q.errorType}</if>" +
            "<if test='q.keyword != null'> AND (t.trace_id LIKE CONCAT('%', #{q.keyword}, '%') " +
            "OR c.title LIKE CONCAT('%', #{q.keyword}, '%'))</if>" +
            "<if test='q.capability != null'> AND JSON_SEARCH(t.context_snapshot, 'one', #{q.capability}, NULL, '$.items[*].type') IS NOT NULL</if>" +
            "<if test='start != null'> AND t.create_time &gt;= #{start}</if>" +
            "<if test='end != null'> AND t.create_time &lt;= #{end}</if>" +
            " ORDER BY t.create_time DESC, t.id DESC LIMIT #{offset}, #{limit}</script>")
    List<SysAiTrace> selectTracePage(@Param("q") AiObservabilityTraceQuery query,
                                     @Param("start") LocalDateTime start,
                                     @Param("end") LocalDateTime end,
                                     @Param("offset") long offset,
                                     @Param("limit") long limit);

    /** 过程链导出：同检索条件全量取数（行数上限由 service 先 count 校验） */
    @Select("<script>SELECT t.* FROM sys_ai_trace t " +
            "<if test='q.userId != null or q.keyword != null'>JOIN sys_ai_conversation c ON c.id = t.conversation_id </if>" +
            "WHERE 1 = 1 " +
            "<if test='q.conversationId != null'> AND t.conversation_id = #{q.conversationId}</if>" +
            "<if test='q.userId != null'> AND c.user_id = #{q.userId} AND c.deleted = 0</if>" +
            "<if test='q.status != null'> AND t.status = #{q.status}</if>" +
            "<if test='q.agentCode != null'> AND t.agent_code = #{q.agentCode}</if>" +
            "<if test='q.model != null'> AND t.model = #{q.model}</if>" +
            "<if test='q.errorType != null'> AND t.error_type = #{q.errorType}</if>" +
            "<if test='q.keyword != null'> AND (t.trace_id LIKE CONCAT('%', #{q.keyword}, '%') " +
            "OR c.title LIKE CONCAT('%', #{q.keyword}, '%'))</if>" +
            "<if test='q.capability != null'> AND JSON_SEARCH(t.context_snapshot, 'one', #{q.capability}, NULL, '$.items[*].type') IS NOT NULL</if>" +
            "<if test='start != null'> AND t.create_time &gt;= #{start}</if>" +
            "<if test='end != null'> AND t.create_time &lt;= #{end}</if>" +
            " ORDER BY t.create_time DESC, t.id DESC</script>")
    List<SysAiTrace> selectTraces(@Param("q") AiObservabilityTraceQuery query,
                                  @Param("start") LocalDateTime start,
                                  @Param("end") LocalDateTime end);

    /** 消耗聚合分组数（分页总数） */
    @Select("<script>SELECT COUNT(*) FROM (SELECT 1 FROM sys_ai_trace t " +
            "<if test=\"dimension == 'user'\">JOIN sys_ai_conversation c ON c.id = t.conversation_id </if>" +
            "WHERE 1 = 1 " +
            "<if test='start != null'> AND t.create_time &gt;= #{start}</if>" +
            "<if test='end != null'> AND t.create_time &lt;= #{end}</if>" +
            " GROUP BY " +
            "<choose><when test=\"dimension == 'model'\">t.model</when>" +
            "<when test=\"dimension == 'agent'\">t.agent_code</when>" +
            "<otherwise>c.user_id</otherwise></choose>" +
            ") grouped</script>")
    long countCostGroups(@Param("dimension") String dimension,
                         @Param("start") LocalDateTime start,
                         @Param("end") LocalDateTime end);

    @Select("<script>SELECT " +
            "<choose><when test=\"dimension == 'model'\">t.model</when>" +
            "<when test=\"dimension == 'agent'\">t.agent_code</when>" +
            "<otherwise>c.user_id</otherwise></choose> AS dimension, " +
            "COUNT(*) AS traceCount, " +
            "COALESCE(SUM(t.total_tokens), 0) AS totalTokens, " +
            "COALESCE(SUM(t.prompt_tokens), 0) AS promptTokens, " +
            "COALESCE(SUM(t.completion_tokens), 0) AS completionTokens, " +
            "COALESCE(SUM(t.cached_tokens), 0) AS cachedTokens " +
            "FROM sys_ai_trace t " +
            "<if test=\"dimension == 'user'\">JOIN sys_ai_conversation c ON c.id = t.conversation_id </if>" +
            "WHERE 1 = 1 " +
            "<if test='start != null'> AND t.create_time &gt;= #{start}</if>" +
            "<if test='end != null'> AND t.create_time &lt;= #{end}</if>" +
            " GROUP BY " +
            "<choose><when test=\"dimension == 'model'\">t.model</when>" +
            "<when test=\"dimension == 'agent'\">t.agent_code</when>" +
            "<otherwise>c.user_id</otherwise></choose>" +
            " ORDER BY dimension LIMIT #{offset}, #{limit}</script>")
    List<AiTraceCostRead> selectCostRows(@Param("dimension") String dimension,
                                         @Param("start") LocalDateTime start,
                                         @Param("end") LocalDateTime end,
                                         @Param("offset") long offset,
                                         @Param("limit") long limit);

    @Select("<script>SELECT DATE_FORMAT(t.create_time, '%Y-%m-%d') AS date, " +
            "COUNT(*) AS traceCount, " +
            "COALESCE(SUM(t.total_tokens), 0) AS totalTokens, " +
            "COALESCE(SUM(t.prompt_tokens), 0) AS promptTokens, " +
            "COALESCE(SUM(t.completion_tokens), 0) AS completionTokens, " +
            "COALESCE(SUM(t.cached_tokens), 0) AS cachedTokens " +
            "FROM sys_ai_trace t " +
            "<if test=\"dimension == 'user'\">JOIN sys_ai_conversation c ON c.id = t.conversation_id </if>" +
            "WHERE 1 = 1 " +
            "<if test='start != null'> AND t.create_time &gt;= #{start}</if>" +
            "<if test='end != null'> AND t.create_time &lt;= #{end}</if>" +
            " GROUP BY date ORDER BY date</script>")
    List<AiTraceCostTrendRead> selectCostTrend(@Param("dimension") String dimension,
                                               @Param("start") LocalDateTime start,
                                               @Param("end") LocalDateTime end);

    @Select("<script>SELECT " +
            "<choose><when test=\"dimension == 'model'\">t.model</when>" +
            "<otherwise>t.agent_code</otherwise></choose> AS dimension, " +
            "DATE_FORMAT(t.create_time, '%Y-%m-%d') AS date, " +
            "COUNT(*) AS callCount, " +
            "SUM(CASE WHEN t.status = 1 THEN 1 ELSE 0 END) AS successCount, " +
            "AVG(CASE WHEN t.status = 1 THEN t.first_token_ms END) AS avgFirstTokenMs, " +
            "AVG(t.duration_ms) AS avgDurationMs " +
            "FROM sys_ai_trace t WHERE 1 = 1 " +
            "<if test='start != null'> AND t.create_time &gt;= #{start}</if>" +
            "<if test='end != null'> AND t.create_time &lt;= #{end}</if>" +
            " GROUP BY " +
            "<choose><when test=\"dimension == 'model'\">t.model</when>" +
            "<otherwise>t.agent_code</otherwise></choose>" +
            ", date ORDER BY date, dimension</script>")
    List<AiTraceTrendRead> selectTrends(@Param("dimension") String dimension,
                                        @Param("start") LocalDateTime start,
                                        @Param("end") LocalDateTime end);
}
