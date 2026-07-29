package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.dehaze.model.entity.SysFeedback;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

@Mapper
public interface SysFeedbackMapper extends BaseMapper<SysFeedback> {

    @Select("<script>" +
            "SELECT COUNT(*) FROM sys_feedback WHERE deleted = 0 " +
            "<if test='startTime != null'> AND create_time &gt;= #{startTime}</if>" +
            "<if test='endTime != null'> AND create_time &lt;= #{endTime}</if>" +
            "</script>")
    long countTotal(@Param("startTime") LocalDateTime startTime, @Param("endTime") LocalDateTime endTime);

    @Select("<script>" +
            "SELECT feedback_type AS feedbackType, COUNT(*) AS cnt FROM sys_feedback WHERE deleted = 0 " +
            "<if test='startTime != null'> AND create_time &gt;= #{startTime}</if>" +
            "<if test='endTime != null'> AND create_time &lt;= #{endTime}</if>" +
            " GROUP BY feedback_type" +
            "</script>")
    List<Map<String, Object>> selectTypeDistribution(@Param("startTime") LocalDateTime startTime,
                                                     @Param("endTime") LocalDateTime endTime);

    @Select("<script>" +
            "SELECT status, COUNT(*) AS cnt FROM sys_feedback WHERE deleted = 0 " +
            "<if test='startTime != null'> AND create_time &gt;= #{startTime}</if>" +
            "<if test='endTime != null'> AND create_time &lt;= #{endTime}</if>" +
            " GROUP BY status" +
            "</script>")
    List<Map<String, Object>> selectStatusDistribution(@Param("startTime") LocalDateTime startTime,
                                                       @Param("endTime") LocalDateTime endTime);

    @Select("<script>" +
            "SELECT related_module AS relatedModule, COUNT(*) AS cnt FROM sys_feedback WHERE deleted = 0 AND related_module IS NOT NULL AND related_module != '' " +
            "<if test='startTime != null'> AND create_time &gt;= #{startTime}</if>" +
            "<if test='endTime != null'> AND create_time &lt;= #{endTime}</if>" +
            " GROUP BY related_module" +
            "</script>")
    List<Map<String, Object>> selectModuleDistribution(@Param("startTime") LocalDateTime startTime,
                                                       @Param("endTime") LocalDateTime endTime);

    @Select("<script>" +
            "SELECT id, create_time AS createTime, update_time AS updateTime, status FROM sys_feedback WHERE deleted = 0 " +
            "<if test='startTime != null'> AND create_time &gt;= #{startTime}</if>" +
            "<if test='endTime != null'> AND create_time &lt;= #{endTime}</if>" +
            "</script>")
    List<Map<String, Object>> selectFeedbackTimes(@Param("startTime") LocalDateTime startTime,
                                                  @Param("endTime") LocalDateTime endTime);

    @Select("<script>" +
            "SELECT fr.feedback_id AS feedbackId, MIN(fr.create_time) AS firstReplyTime " +
            "FROM sys_feedback_reply fr " +
            "INNER JOIN sys_feedback f ON fr.feedback_id = f.id AND f.deleted = 0 " +
            "WHERE fr.replier_type = 2 " +
            "<if test='startTime != null'> AND f.create_time &gt;= #{startTime}</if>" +
            "<if test='endTime != null'> AND f.create_time &lt;= #{endTime}</if>" +
            " GROUP BY fr.feedback_id" +
            "</script>")
    List<Map<String, Object>> selectFirstReplyTimes(@Param("startTime") LocalDateTime startTime,
                                                    @Param("endTime") LocalDateTime endTime);

    @Select("<script>" +
            "SELECT title, content FROM sys_feedback WHERE deleted = 0 " +
            "<if test='startTime != null'> AND create_time &gt;= #{startTime}</if>" +
            "<if test='endTime != null'> AND create_time &lt;= #{endTime}</if>" +
            "</script>")
    List<Map<String, Object>> selectTitleAndContent(@Param("startTime") LocalDateTime startTime,
                                                    @Param("endTime") LocalDateTime endTime);
}
