package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.pei.dehaze.model.entity.SysAiAgentEvalRun;
import org.apache.ibatis.annotations.Delete;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.time.LocalDateTime;
import java.util.List;

/**
 * 智能体评测执行记录访问层
 *
 * @author dehaze
 */
@Mapper
public interface SysAiAgentEvalRunMapper extends BaseMapper<SysAiAgentEvalRun> {

    @Delete("DELETE FROM sys_ai_agent_eval_run WHERE agent_id = #{agentId}")
    int deleteByAgentId(@Param("agentId") Long agentId);

    /**
     * 同一 Agent 同一评测集上一次已完成评测（相对退化门禁基准）
     */
    @Select("SELECT * FROM sys_ai_agent_eval_run WHERE agent_id = #{agentId} AND dataset_id = #{datasetId} " +
            "AND status IN (2, 3) AND id != #{excludeRunId} ORDER BY id DESC LIMIT 1")
    SysAiAgentEvalRun selectPreviousCompleted(@Param("agentId") Long agentId, @Param("datasetId") Long datasetId,
                                               @Param("excludeRunId") Long excludeRunId);

    /**
     * 各 Agent 最近 perAgent 次已完成评测（窗口函数，供总览退化判定）
     */
    @Select("SELECT * FROM (SELECT t.*, ROW_NUMBER() OVER (PARTITION BY agent_id ORDER BY id DESC) rn " +
            "FROM sys_ai_agent_eval_run t WHERE t.status IN (2, 3)) x WHERE x.rn <= #{perAgent} " +
            "ORDER BY x.agent_id, x.id DESC")
    List<SysAiAgentEvalRun> selectLatestPerAgent(@Param("perAgent") int perAgent);

    /**
     * 已完成评测（时间升序，供趋势聚合与复核扫描）
     */
    @Select("<script>SELECT * FROM sys_ai_agent_eval_run WHERE status IN (2, 3) " +
            "<if test='agentId != null'> AND agent_id = #{agentId}</if>" +
            "<if test='startTime != null'> AND create_time &gt;= #{startTime}</if>" +
            "<if test='endTime != null'> AND create_time &lt;= #{endTime}</if>" +
            " ORDER BY create_time ASC, id ASC LIMIT #{limit}</script>")
    List<SysAiAgentEvalRun> selectCompleted(@Param("agentId") Long agentId,
                                            @Param("startTime") LocalDateTime startTime,
                                            @Param("endTime") LocalDateTime endTime,
                                            @Param("limit") int limit);

    @Select("<script>SELECT * FROM sys_ai_agent_eval_run WHERE agent_id = #{agentId} " +
            "<if test='datasetId != null'> AND dataset_id = #{datasetId}</if> ORDER BY id DESC</script>")
    IPage<SysAiAgentEvalRun> selectRunPage(IPage<SysAiAgentEvalRun> page, @Param("agentId") Long agentId,
                                           @Param("datasetId") Long datasetId);
}
