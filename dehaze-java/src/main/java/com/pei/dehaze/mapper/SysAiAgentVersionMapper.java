package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.pei.dehaze.model.entity.SysAiAgentVersion;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;

/**
 * AI 智能体版本访问层
 *
 * @author dehaze
 */
@Mapper
public interface SysAiAgentVersionMapper extends BaseMapper<SysAiAgentVersion> {

    /**
     * 下一个版本号（MAX+1；与插入非原子，唯一键冲突由服务层重试）
     */
    @Select("SELECT COALESCE(MAX(version_no), 0) + 1 FROM sys_ai_agent_version WHERE agent_id = #{agentId}")
    int nextVersionNo(@Param("agentId") Long agentId);

    /**
     * 当前已发布版本（status=2，版本号最大）
     */
    @Select("SELECT * FROM sys_ai_agent_version WHERE agent_id = #{agentId} AND status = 2 " +
            "ORDER BY version_no DESC LIMIT 1")
    SysAiAgentVersion getLatestPublished(@Param("agentId") Long agentId);

    /**
     * 最新草稿版本（status=1，发布门禁评测对象）
     */
    @Select("SELECT * FROM sys_ai_agent_version WHERE agent_id = #{agentId} AND status = 1 " +
            "ORDER BY version_no DESC LIMIT 1")
    SysAiAgentVersion getLatestDraft(@Param("agentId") Long agentId);

    /**
     * 指定版本号（唯一键索引生效）
     */
    @Select("SELECT * FROM sys_ai_agent_version WHERE agent_id = #{agentId} AND version_no = #{versionNo}")
    SysAiAgentVersion getByAgentAndVersion(@Param("agentId") Long agentId, @Param("versionNo") Integer versionNo);

    /**
     * 旧已发布版本降级为历史（status=1）
     */
    @Update("UPDATE sys_ai_agent_version SET status = 1 WHERE agent_id = #{agentId} AND status = 2")
    int demotePublished(@Param("agentId") Long agentId);

    /**
     * 版本历史分页（不加载 snapshot 大字段，按版本号倒序）
     */
    @Select("SELECT id, agent_id, version_no, status, change_note, operator_id, create_time " +
            "FROM sys_ai_agent_version WHERE agent_id = #{agentId} ORDER BY version_no DESC")
    IPage<SysAiAgentVersion> selectVersionPage(IPage<SysAiAgentVersion> page, @Param("agentId") Long agentId);
}
