package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.dehaze.model.entity.SysAiEvalReview;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;

/**
 * 人工复核项访问层
 *
 * @author dehaze
 */
@Mapper
public interface SysAiEvalReviewMapper extends BaseMapper<SysAiEvalReview> {

    @Select("<script>SELECT * FROM sys_ai_eval_review WHERE run_id IN " +
            "<foreach collection='runIds' item='id' open='(' separator=',' close=')'>#{id}</foreach></script>")
    List<SysAiEvalReview> listByRunIds(@Param("runIds") List<Long> runIds);

    @Select("SELECT * FROM sys_ai_eval_review ORDER BY status ASC, id DESC LIMIT #{limit}")
    List<SysAiEvalReview> listAll(@Param("limit") int limit);
}
