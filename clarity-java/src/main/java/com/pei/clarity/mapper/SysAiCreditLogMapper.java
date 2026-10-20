package com.pei.clarity.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.clarity.model.entity.SysAiCreditLog;
import java.time.LocalDateTime;
import java.util.List;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

@Mapper
public interface SysAiCreditLogMapper extends BaseMapper<SysAiCreditLog> {

    /**
     * 指定时间区间内产生过某来源流水的用户（月末清零任务筛选 vip_gift 赠送用户用）
     */
    @Select("SELECT DISTINCT user_id FROM sys_ai_credit_log "
            + "WHERE source = #{source} AND deleted = 0 "
            + "AND create_time >= #{start} AND create_time <= #{end}")
    List<Long> selectDistinctUserIdsBySource(
            @Param("source") String source, @Param("start") LocalDateTime start, @Param("end") LocalDateTime end);

    /**
     * 指定用户某来源在时间区间内的流水合计（正负相抵，清零额以此为上限）
     */
    @Select("SELECT COALESCE(SUM(amount), 0) FROM sys_ai_credit_log "
            + "WHERE user_id = #{userId} AND source = #{source} AND deleted = 0 "
            + "AND create_time >= #{start} AND create_time <= #{end}")
    Long sumAmountByUserAndSource(
            @Param("userId") Long userId,
            @Param("source") String source,
            @Param("start") LocalDateTime start,
            @Param("end") LocalDateTime end);
}
