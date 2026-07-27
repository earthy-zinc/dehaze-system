package com.pei.dehaze.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.dehaze.model.entity.SysNotificationSetting;
import com.pei.dehaze.model.form.NotificationSettingForm;
import com.pei.dehaze.model.vo.NotificationSettingsVO;

public interface NotificationSettingService extends IService<SysNotificationSetting> {

    NotificationSettingsVO get();

    void update(NotificationSettingForm form);
}
