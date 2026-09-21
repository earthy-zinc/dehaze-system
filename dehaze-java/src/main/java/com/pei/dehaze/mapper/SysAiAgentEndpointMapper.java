package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.dehaze.model.entity.SysAiAgentEndpoint;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;

import java.util.List;

/**
 * 外部 A2A 端点访问层
 *
 * @author dehaze
 */
@Mapper
public interface SysAiAgentEndpointMapper extends BaseMapper<SysAiAgentEndpoint> {

    /**
     * 按 base_url 查询（含软删行：软删行占用唯一键 uk_base_url(base_url, deleted)）
     */
    @Select("SELECT * FROM sys_ai_agent_endpoint WHERE base_url = #{baseUrl} LIMIT 1")
    SysAiAgentEndpoint selectByBaseUrlIgnoringDeleted(@Param("baseUrl") String baseUrl);

    @Update("<script>UPDATE sys_ai_agent_endpoint SET deleted = id " +
            "WHERE deleted = 0 AND id IN <foreach collection='ids' item='id' open='(' separator=',' close=')'>#{id}</foreach></script>")
    int softDeleteByIds(@Param("ids") List<Long> ids);
}
