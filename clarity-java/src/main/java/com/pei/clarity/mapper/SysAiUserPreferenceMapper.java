package com.pei.clarity.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.clarity.model.entity.SysAiUserPreference;
import org.apache.ibatis.annotations.Insert;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

@Mapper
public interface SysAiUserPreferenceMapper extends BaseMapper<SysAiUserPreference> {

    /**
     * upsert 偏好默认模型：user_id 唯一键冲突时改写既有行。
     *
     * <p>UPDATE 分支必须显式写 default_model——MyBatis-Plus 的 updateById 会忽略 null 字段，
     * 无法把已设的偏好清空（null 即"清除偏好"）。
     */
    @Insert(
            "INSERT INTO sys_ai_user_preference (user_id, default_model, deleted, create_by, update_by, create_time, update_time) "
                    + "VALUES (#{userId}, #{defaultModel}, 0, #{userId}, #{userId}, NOW(), NOW()) "
                    + "ON DUPLICATE KEY UPDATE default_model = #{defaultModel}, update_by = #{userId}, update_time = NOW()")
    int upsertDefaultModel(@Param("userId") Long userId, @Param("defaultModel") String defaultModel);
}
