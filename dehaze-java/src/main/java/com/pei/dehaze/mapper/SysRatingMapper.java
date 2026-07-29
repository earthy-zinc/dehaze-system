package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.dehaze.model.entity.SysRating;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

@Mapper
public interface SysRatingMapper extends BaseMapper<SysRating> {

    @Select("<script>" +
            "SELECT rating, COUNT(*) AS cnt FROM sys_rating WHERE deleted = 0 " +
            "<if test='startTime != null'> AND create_time &gt;= #{startTime}</if>" +
            "<if test='endTime != null'> AND create_time &lt;= #{endTime}</if>" +
            " GROUP BY rating" +
            "</script>")
    List<Map<String, Object>> selectRatingDistribution(@Param("startTime") LocalDateTime startTime,
                                                       @Param("endTime") LocalDateTime endTime);

    @Select("<script>" +
            "SELECT algorithm_id AS algorithmId, a.name AS algorithmName, " +
            "AVG(r.rating) AS avgRating, COUNT(*) AS total, " +
            "SUM(CASE WHEN r.rating &lt;= 2 THEN 1 ELSE 0 END) AS lowCount " +
            "FROM sys_rating r LEFT JOIN sys_algorithm a ON r.algorithm_id = a.id " +
            "WHERE r.deleted = 0 " +
            "<if test='startTime != null'> AND r.create_time &gt;= #{startTime}</if>" +
            "<if test='endTime != null'> AND r.create_time &lt;= #{endTime}</if>" +
            " GROUP BY r.algorithm_id, a.name" +
            "</script>")
    List<Map<String, Object>> selectAlgorithmStats(@Param("startTime") LocalDateTime startTime,
                                                   @Param("endTime") LocalDateTime endTime);

    @Select("<script>" +
            "SELECT tags FROM sys_rating WHERE deleted = 0 AND tags IS NOT NULL AND tags != '' AND tags != '[]' " +
            "<if test='startTime != null'> AND create_time &gt;= #{startTime}</if>" +
            "<if test='endTime != null'> AND create_time &lt;= #{endTime}</if>" +
            "</script>")
    List<String> selectAllTags(@Param("startTime") LocalDateTime startTime, @Param("endTime") LocalDateTime endTime);

    @Select("SELECT COUNT(*) FROM sys_rating WHERE deleted = 0 AND rating <= #{threshold} AND create_time >= #{startTime}")
    long countLowRatingsSince(@Param("threshold") int threshold, @Param("startTime") LocalDateTime startTime);

    @Select("SELECT COUNT(*) FROM sys_rating WHERE deleted = 0 AND create_time >= #{startTime}")
    long countRatingsSince(@Param("startTime") LocalDateTime startTime);
}
