package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.dehaze.model.entity.SysRecommendation;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

@Mapper
public interface SysRecommendationMapper extends BaseMapper<SysRecommendation> {

    @Select("<script>" +
            "SELECT COUNT(*) FROM sys_recommendation WHERE 1=1 " +
            "<if test='startDate != null'> AND create_time &gt;= #{startDate}</if>" +
            "<if test='endDate != null'> AND create_time &lt;= #{endDate}</if>" +
            "</script>")
    long countTotal(@Param("startDate") LocalDateTime startDate, @Param("endDate") LocalDateTime endDate);

    @Select("<script>" +
            "SELECT COUNT(*) FROM sys_recommendation WHERE feedback = 1 " +
            "<if test='startDate != null'> AND create_time &gt;= #{startDate}</if>" +
            "<if test='endDate != null'> AND create_time &lt;= #{endDate}</if>" +
            "</script>")
    long countUseful(@Param("startDate") LocalDateTime startDate, @Param("endDate") LocalDateTime endDate);

    @Select("<script>" +
            "SELECT COUNT(*) FROM sys_recommendation WHERE feedback IN (1, 2) " +
            "<if test='startDate != null'> AND create_time &gt;= #{startDate}</if>" +
            "<if test='endDate != null'> AND create_time &lt;= #{endDate}</if>" +
            "</script>")
    long countFeedbackTotal(@Param("startDate") LocalDateTime startDate, @Param("endDate") LocalDateTime endDate);

    @Select("<script>" +
            "SELECT COUNT(DISTINCT adopted_algorithm_id) FROM sys_recommendation WHERE adopted_algorithm_id IS NOT NULL " +
            "<if test='startDate != null'> AND create_time &gt;= #{startDate}</if>" +
            "<if test='endDate != null'> AND create_time &lt;= #{endDate}</if>" +
            "</script>")
    long countAdoptedAlgorithmDistinct(@Param("startDate") LocalDateTime startDate, @Param("endDate") LocalDateTime endDate);

    @Select("<script>" +
            "SELECT DATE(create_time) AS date, " +
            "CASE WHEN COUNT(*) > 0 THEN CAST(SUM(CASE WHEN feedback = 1 THEN 1 ELSE 0 END) AS DECIMAL(10,4)) / COUNT(*) ELSE 0 END AS adoptionRate " +
            "FROM sys_recommendation WHERE feedback IN (1, 2) " +
            "<if test='startDate != null'> AND create_time &gt;= #{startDate}</if>" +
            "<if test='endDate != null'> AND create_time &lt;= #{endDate}</if>" +
            " GROUP BY DATE(create_time) ORDER BY date" +
            "</script>")
    List<Map<String, Object>> selectDailyAdoptionRate(@Param("startDate") LocalDateTime startDate,
                                                       @Param("endDate") LocalDateTime endDate);

    @Select("SELECT COUNT(*) FROM sys_recommendation WHERE feedback IN (1, 2) AND adopted_algorithm_id IS NOT NULL")
    long countAdoptedWithFeedback();
}
