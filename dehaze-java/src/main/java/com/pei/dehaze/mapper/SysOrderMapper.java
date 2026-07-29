package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.dehaze.model.entity.SysOrder;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

@Mapper
public interface SysOrderMapper extends BaseMapper<SysOrder> {

    @Select("<script>" +
            "SELECT COUNT(*) FROM sys_order WHERE deleted = 0 " +
            "<if test='startTime != null'> AND create_time &gt;= #{startTime}</if>" +
            "<if test='endTime != null'> AND create_time &lt;= #{endTime}</if>" +
            "</script>")
    long countTotalOrders(@Param("startTime") LocalDateTime startTime, @Param("endTime") LocalDateTime endTime);

    @Select("<script>" +
            "SELECT COALESCE(SUM(paid_amount), 0) FROM sys_order WHERE deleted = 0 AND status IN (2, 3) " +
            "<if test='startTime != null'> AND create_time &gt;= #{startTime}</if>" +
            "<if test='endTime != null'> AND create_time &lt;= #{endTime}</if>" +
            "</script>")
    long sumRevenue(@Param("startTime") LocalDateTime startTime, @Param("endTime") LocalDateTime endTime);

    @Select("<script>" +
            "SELECT status, COUNT(*) AS cnt FROM sys_order WHERE deleted = 0 " +
            "<if test='startTime != null'> AND create_time &gt;= #{startTime}</if>" +
            "<if test='endTime != null'> AND create_time &lt;= #{endTime}</if>" +
            " GROUP BY status" +
            "</script>")
    List<Map<String, Object>> selectStatusDistribution(@Param("startTime") LocalDateTime startTime, @Param("endTime") LocalDateTime endTime);

    @Select("<script>" +
            "SELECT pay_method AS payMethod, COUNT(*) AS cnt FROM sys_order WHERE deleted = 0 AND status IN (2, 3) " +
            "<if test='startTime != null'> AND create_time &gt;= #{startTime}</if>" +
            "<if test='endTime != null'> AND create_time &lt;= #{endTime}</if>" +
            " GROUP BY pay_method" +
            "</script>")
    List<Map<String, Object>> selectPayMethodDistribution(@Param("startTime") LocalDateTime startTime, @Param("endTime") LocalDateTime endTime);

    @Select("<script>" +
            "SELECT package_id AS packageId, package_name AS packageName, COUNT(*) AS cnt, COALESCE(SUM(paid_amount), 0) AS revenue " +
            "FROM sys_order WHERE deleted = 0 AND status IN (2, 3) " +
            "<if test='startTime != null'> AND create_time &gt;= #{startTime}</if>" +
            "<if test='endTime != null'> AND create_time &lt;= #{endTime}</if>" +
            " GROUP BY package_id, package_name" +
            "</script>")
    List<Map<String, Object>> selectPackageDistribution(@Param("startTime") LocalDateTime startTime, @Param("endTime") LocalDateTime endTime);

    @Select("<script>" +
            "SELECT DATE_FORMAT(create_time, '%Y-%m-%d') AS date, COUNT(*) AS cnt, COALESCE(SUM(paid_amount), 0) AS revenue " +
            "FROM sys_order WHERE deleted = 0 AND status IN (2, 3) " +
            "<if test='startTime != null'> AND create_time &gt;= #{startTime}</if>" +
            "<if test='endTime != null'> AND create_time &lt;= #{endTime}</if>" +
            " GROUP BY DATE_FORMAT(create_time, '%Y-%m-%d') ORDER BY date DESC LIMIT 30" +
            "</script>")
    List<Map<String, Object>> selectDailyStats(@Param("startTime") LocalDateTime startTime, @Param("endTime") LocalDateTime endTime);
}
