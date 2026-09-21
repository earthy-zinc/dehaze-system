package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.dehaze.model.entity.SysAiMcpServer;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;

@Mapper
public interface SysAiMcpServerMapper extends BaseMapper<SysAiMcpServer> {

    /**
     * 按名称查询（含软删行）。
     *
     * <p>DB 唯一键 name 含软删行，市场一键接入需复活同名软删 Server 而非新建。
     */
    @Select("SELECT * FROM sys_ai_mcp_server WHERE name = #{name} LIMIT 1")
    SysAiMcpServer selectByNameIncludingDeleted(@Param("name") String name);

    /** 复活软删行（市场一键接入：重置删除标记并置启用） */
    @Update("UPDATE sys_ai_mcp_server SET deleted = 0, status = 1, update_time = NOW() WHERE id = #{id}")
    int resurrect(@Param("id") Long id);

    /**
     * 统计关联了该 Server 命名空间的 Agent 数。
     *
     * <p>直接读 sys_ai_agent_mcp / sys_ai_mcp_namespace（同义命名空间跨 Server 复用时保守计入）。
     */
    @Select("SELECT COUNT(DISTINCT am.agent_id) FROM sys_ai_agent_mcp am "
            + "JOIN sys_ai_mcp_namespace ns ON ns.namespace = am.mcp_namespace "
            + "WHERE ns.server_id = #{serverId}")
    long countAgentReferences(@Param("serverId") Long serverId);
}
