package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.dehaze.model.entity.SysRefundRecord;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

@Mapper
public interface SysRefundRecordMapper extends BaseMapper<SysRefundRecord> {

    @Select("<script>" +
            "SELECT COALESCE(SUM(refund_amount), 0) FROM sys_refund_record WHERE deleted = 0 AND status = 2 " +
            "<if test='startTime != null'> AND refund_time &gt;= #{startTime}</if>" +
            "<if test='endTime != null'> AND refund_time &lt;= #{endTime}</if>" +
            "</script>")
    long sumRefundAmount(@Param("startTime") LocalDateTime startTime, @Param("endTime") LocalDateTime endTime);

    @Select("<script>" +
            "SELECT r.*, o.order_no AS orderNo, u.username AS username " +
            "FROM sys_refund_record r " +
            "LEFT JOIN sys_order o ON r.order_id = o.id " +
            "LEFT JOIN sys_user u ON r.user_id = u.id " +
            "WHERE r.deleted = 0 " +
            "<if test='status != null'> AND r.status = #{status}</if>" +
            "<if test='applyTimeStart != null'> AND r.apply_time &gt;= #{applyTimeStart}</if>" +
            "<if test='applyTimeEnd != null'> AND r.apply_time &lt;= #{applyTimeEnd}</if>" +
            "<if test='orderNo != null and orderNo != \"\"'> AND o.order_no LIKE CONCAT('%', #{orderNo}, '%')</if>" +
            "<if test='keywords != null and keywords != \"\"'> AND (u.username LIKE CONCAT('%', #{keywords}, '%') OR u.nickname LIKE CONCAT('%', #{keywords}, '%'))</if>" +
            " ORDER BY r.id DESC LIMIT #{offset}, #{limit}" +
            "</script>")
    List<Map<String, Object>> selectRefundPageWithLimit(@Param("status") Integer status,
                                                        @Param("applyTimeStart") LocalDateTime applyTimeStart,
                                                        @Param("applyTimeEnd") LocalDateTime applyTimeEnd,
                                                        @Param("orderNo") String orderNo,
                                                        @Param("keywords") String keywords,
                                                        @Param("offset") int offset,
                                                        @Param("limit") int limit);

    @Select("<script>" +
            "SELECT COUNT(*) FROM sys_refund_record r " +
            "LEFT JOIN sys_order o ON r.order_id = o.id " +
            "LEFT JOIN sys_user u ON r.user_id = u.id " +
            "WHERE r.deleted = 0 " +
            "<if test='status != null'> AND r.status = #{status}</if>" +
            "<if test='applyTimeStart != null'> AND r.apply_time &gt;= #{applyTimeStart}</if>" +
            "<if test='applyTimeEnd != null'> AND r.apply_time &lt;= #{applyTimeEnd}</if>" +
            "<if test='orderNo != null and orderNo != \"\"'> AND o.order_no LIKE CONCAT('%', #{orderNo}, '%')</if>" +
            "<if test='keywords != null and keywords != \"\"'> AND (u.username LIKE CONCAT('%', #{keywords}, '%') OR u.nickname LIKE CONCAT('%', #{keywords}, '%'))</if>" +
            "</script>")
    long countRefundPage(@Param("status") Integer status,
                         @Param("applyTimeStart") LocalDateTime applyTimeStart,
                         @Param("applyTimeEnd") LocalDateTime applyTimeEnd,
                         @Param("orderNo") String orderNo,
                         @Param("keywords") String keywords);
}
