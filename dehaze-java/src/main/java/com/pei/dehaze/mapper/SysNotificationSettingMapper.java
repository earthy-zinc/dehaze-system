package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.dehaze.model.entity.SysNotificationSetting;
import org.apache.ibatis.annotations.Insert;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

@Mapper
public interface SysNotificationSettingMapper extends BaseMapper<SysNotificationSetting> {

    /**
     * upsert 通知设置：user_id 唯一键冲突时复活软删行。
     * UPDATE 分支通过 LAST_INSERT_ID(id) 拿回原行 id。
     */
    @Insert("INSERT INTO sys_notification_setting (user_id, push_enabled, email_enabled, sms_enabled, deleted, update_time) " +
            "VALUES (#{userId}, COALESCE(#{pushEnabled}, 1), COALESCE(#{emailEnabled}, 1), COALESCE(#{smsEnabled}, 0), 0, NOW()) " +
            "ON DUPLICATE KEY UPDATE id = LAST_INSERT_ID(id), deleted = 0, " +
            "push_enabled = COALESCE(#{pushEnabled}, push_enabled), " +
            "email_enabled = COALESCE(#{emailEnabled}, email_enabled), " +
            "sms_enabled = COALESCE(#{smsEnabled}, sms_enabled), update_time = NOW()")
    int upsertByUser(@Param("userId") Long userId,
                     @Param("pushEnabled") Boolean pushEnabled,
                     @Param("emailEnabled") Boolean emailEnabled,
                     @Param("smsEnabled") Boolean smsEnabled);
}
