package com.pei.dehaze.mapper;

import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.Map;

/**
 * A2A Agent Card 只读查询。
 *
 * <p>Agent 主体与版本实体归属 AI 对话模块，此处仅取 Card 所需的几个标量字段，
 * 避免跨人重复定义实体。
 */
@Mapper
public interface AiA2aMapper {

    /** 按主键取 Agent 的对外服务判定字段（status/is_exposed/is_subagent）与展示字段 */
    @Select("SELECT id, name, agent_code AS agentCode, description, status, "
            + "is_exposed AS isExposed, is_subagent AS isSubagent "
            + "FROM sys_ai_agent WHERE id = #{agentId}")
    Map<String, Object> selectAgentForCard(@Param("agentId") Long agentId);

    /** Agent 已发布版本号（无已发布版本时返回 null） */
    @Select("SELECT version_no FROM sys_ai_agent_version WHERE agent_id = #{agentId} "
            + "AND status = 2 ORDER BY version_no DESC LIMIT 1")
    Integer selectPublishedVersionNo(@Param("agentId") Long agentId);
}
