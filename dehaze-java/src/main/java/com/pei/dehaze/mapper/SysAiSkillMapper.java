package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.dehaze.model.entity.SysAiSkill;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;
import java.util.Map;

@Mapper
public interface SysAiSkillMapper extends BaseMapper<SysAiSkill> {

    /** 统计关联了该 Skill 名称的 Agent 数（关联表无软删列，删除前校验用） */
    @Select("SELECT COUNT(*) FROM sys_ai_agent_skill WHERE skill_name = #{skillName}")
    long countAgentReferences(@Param("skillName") String skillName);

    /** 批量统计 Agent 关联数（列表装配，避免逐条子查询） */
    @Select("""
            <script>
            SELECT skill_name AS skillName, COUNT(*) AS total
            FROM sys_ai_agent_skill
            WHERE skill_name IN
            <foreach collection="names" item="name" open="(" separator="," close=")">#{name}</foreach>
            GROUP BY skill_name
            </script>
            """)
    List<Map<String, Object>> countAgentReferencesByNames(@Param("names") List<String> names);
}
