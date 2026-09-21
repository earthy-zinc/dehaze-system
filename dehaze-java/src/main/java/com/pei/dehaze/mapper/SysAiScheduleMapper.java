package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.dehaze.model.entity.SysAiSchedule;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;

import java.time.LocalDateTime;

/**
 * AI 定时任务访问层
 *
 * @author dehaze
 */
@Mapper
public interface SysAiScheduleMapper extends BaseMapper<SysAiSchedule> {

    @Select("SELECT COUNT(*) FROM sys_ai_schedule WHERE user_id = #{userId} AND deleted = 0")
    long countByUserId(@Param("userId") Long userId);

    @Update("UPDATE sys_ai_schedule SET enabled = #{enabled} WHERE id = #{scheduleId}")
    int setEnabled(@Param("scheduleId") Long scheduleId, @Param("enabled") Integer enabled);

    /**
     * 熔断恢复：清零连续失败计数并置回正常状态
     */
    @Update("UPDATE sys_ai_schedule SET circuit_streak = 0, status = 1 WHERE id = #{scheduleId}")
    int resetCircuit(@Param("scheduleId") Long scheduleId);

    @Update("UPDATE sys_ai_schedule SET next_trigger_time = #{nextTriggerTime} WHERE id = #{scheduleId}")
    int updateNextTrigger(@Param("scheduleId") Long scheduleId, @Param("nextTriggerTime") LocalDateTime nextTriggerTime);

    @Update("UPDATE sys_ai_schedule SET deleted = id WHERE id = #{scheduleId} AND deleted = 0")
    int softDelete(@Param("scheduleId") Long scheduleId);
}
