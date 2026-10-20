package com.pei.clarity.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.clarity.model.entity.SysAiAgentEvalSample;
import java.util.List;
import org.apache.ibatis.annotations.Delete;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

/**
 * 智能体评测样本访问层（物理删除，随评测集清理）
 *
 * @author clarity
 */
@Mapper
public interface SysAiAgentEvalSampleMapper extends BaseMapper<SysAiAgentEvalSample> {

    @Delete("<script>DELETE FROM sys_ai_agent_eval_sample WHERE dataset_id IN "
            + "<foreach collection='datasetIds' item='id' open='(' separator=',' close=')'>#{id}</foreach></script>")
    int deleteByDatasetIds(@Param("datasetIds") List<Long> datasetIds);
}
