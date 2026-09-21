package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.dehaze.model.entity.SysPredLog;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

@Mapper
public interface SysPredLogMapper extends BaseMapper<SysPredLog> {

    @Select("<script>" +
            "SELECT DATE(create_time) AS date, COUNT(*) AS cnt FROM sys_pred_log WHERE recommended_by IS NOT NULL " +
            "<if test='startDate != null'> AND create_time &gt;= #{startDate}</if>" +
            "<if test='endDate != null'> AND create_time &lt;= #{endDate}</if>" +
            " GROUP BY DATE(create_time) ORDER BY date" +
            "</script>")
    List<Map<String, Object>> selectDailyRecommended(@Param("startDate") LocalDateTime startDate,
                                                     @Param("endDate") LocalDateTime endDate);
}
