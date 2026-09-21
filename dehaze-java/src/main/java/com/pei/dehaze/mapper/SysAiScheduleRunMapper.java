package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.pei.dehaze.model.entity.SysAiScheduleRun;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;

/**
 * AI 定时任务执行历史访问层
 *
 * @author dehaze
 */
@Mapper
public interface SysAiScheduleRunMapper extends BaseMapper<SysAiScheduleRun> {

    /**
     * 各定时任务最近一次执行（列表聚合，避免 N+1）
     */
    @Select("<script>SELECT * FROM (SELECT r.*, ROW_NUMBER() OVER (PARTITION BY schedule_id ORDER BY id DESC) rn " +
            "FROM sys_ai_schedule_run r WHERE r.schedule_id IN " +
            "<foreach collection='scheduleIds' item='id' open='(' separator=',' close=')'>#{id}</foreach>) x " +
            "WHERE x.rn = 1</script>")
    List<SysAiScheduleRun> selectLatestByScheduleIds(@Param("scheduleIds") List<Long> scheduleIds);

    @Select("SELECT * FROM sys_ai_schedule_run WHERE schedule_id = #{scheduleId} ORDER BY id DESC")
    IPage<SysAiScheduleRun> selectPageBySchedule(IPage<SysAiScheduleRun> page, @Param("scheduleId") Long scheduleId);
}
