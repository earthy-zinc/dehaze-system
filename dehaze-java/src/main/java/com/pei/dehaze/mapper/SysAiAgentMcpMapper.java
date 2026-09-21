package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.dehaze.model.entity.SysAiAgentMcp;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;

/**
 * 智能体-MCP 命名空间关联访问层
 *
 * @author dehaze
 */
@Mapper
public interface SysAiAgentMcpMapper extends BaseMapper<SysAiAgentMcp> {

    /**
     * 存在性校验：返回已注册 MCP Server 下声明的命名空间
     */
    @Select("<script>SELECT DISTINCT namespace FROM sys_ai_mcp_namespace WHERE namespace IN " +
            "<foreach collection='names' item='n' open='(' separator=',' close=')'>#{n}</foreach></script>")
    List<String> listRegisteredNamespaces(@Param("names") List<String> names);
}
