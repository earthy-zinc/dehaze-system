package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.dehaze.model.entity.SysAiAgentEvalDataset;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;

import java.util.List;

/**
 * 智能体评测集访问层
 *
 * @author dehaze
 */
@Mapper
public interface SysAiAgentEvalDatasetMapper extends BaseMapper<SysAiAgentEvalDataset> {

    /**
     * 按 (agent_id, dataset_type) 查询（含软删行：软删行占用唯一键，重建时复活原行）
     */
    @Select("SELECT * FROM sys_ai_agent_eval_dataset WHERE agent_id = #{agentId} AND dataset_type = #{datasetType} LIMIT 1")
    SysAiAgentEvalDataset selectByAgentAndTypeIgnoringDeleted(@Param("agentId") Long agentId,
                                                              @Param("datasetType") String datasetType);

    /**
     * 软删（deleted 置行 id，释放唯一键位）
     */
    @Update("<script>UPDATE sys_ai_agent_eval_dataset SET deleted = id " +
            "WHERE deleted = 0 AND id IN <foreach collection='ids' item='id' open='(' separator=',' close=')'>#{id}</foreach></script>")
    int softDeleteByIds(@Param("ids") List<Long> ids);

    /**
     * 复活软删行（配合唯一键 (agent_id, dataset_type, deleted)）
     */
    @Update("UPDATE sys_ai_agent_eval_dataset SET deleted = 0, name = #{name}, description = #{description} " +
            "WHERE id = #{id}")
    int revive(@Param("id") Long id, @Param("name") String name, @Param("description") String description);
}
