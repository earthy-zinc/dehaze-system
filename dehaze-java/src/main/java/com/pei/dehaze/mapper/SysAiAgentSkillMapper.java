package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.dehaze.model.entity.SysAiAgentSkill;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;

/**
 * 智能体-Skill 关联访问层
 *
 * @author dehaze
 */
@Mapper
public interface SysAiAgentSkillMapper extends BaseMapper<SysAiAgentSkill> {

    /**
     * 存在性校验：返回 sys_ai_skill 中真实存在的名称（未删除）
     */
    @Select("<script>SELECT name FROM sys_ai_skill WHERE deleted = 0 AND name IN " +
            "<foreach collection='names' item='n' open='(' separator=',' close=')'>#{n}</foreach></script>")
    List<String> listExistingSkillNames(@Param("names") List<String> names);
}
