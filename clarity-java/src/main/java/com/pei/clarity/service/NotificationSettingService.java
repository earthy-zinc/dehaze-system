package com.pei.clarity.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.clarity.model.entity.SysNotificationSetting;
import com.pei.clarity.model.form.NotificationSettingForm;
import com.pei.clarity.model.vo.NotificationSettingsVO;

public interface NotificationSettingService extends IService<SysNotificationSetting> {

    NotificationSettingsVO get();

    void update(NotificationSettingForm form);
}
