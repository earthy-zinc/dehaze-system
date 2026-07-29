package com.pei.dehaze.service.impl;

import cn.hutool.core.text.CharSequenceUtil;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.mapper.SysNotificationSettingMapper;
import com.pei.dehaze.model.entity.SysNotificationSetting;
import com.pei.dehaze.model.form.NotificationSettingForm;
import com.pei.dehaze.model.vo.NotificationSettingsVO;
import com.pei.dehaze.security.util.SecurityUtils;
import com.pei.dehaze.service.NotificationSettingService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.time.LocalTime;
import java.time.format.DateTimeFormatter;

@Service
@RequiredArgsConstructor
public class NotificationSettingServiceImpl extends ServiceImpl<SysNotificationSettingMapper, SysNotificationSetting> implements NotificationSettingService {

    private static final DateTimeFormatter TIME_FORMATTER = DateTimeFormatter.ofPattern("HH:mm:ss");
    private static final String DEFAULT_PREFERENCES = "{\"typeChannels\":{\"announcement\":{\"push\":true},\"business\":{\"push\":false},\"member\":{\"push\":true}},\"moduleSwitches\":{\"prediction\":true,\"feedback\":true,\"announcement\":true}}";

    @Override
    public NotificationSettingsVO get() {
        Long userId = SecurityUtils.getUserId();
        SysNotificationSetting setting = getOrCreateDefault(userId);
        return toVO(setting);
    }

    @Override
    public void update(NotificationSettingForm form) {
        Long userId = SecurityUtils.getUserId();
        SysNotificationSetting setting = getOrCreateDefault(userId);

        if (form.getPushEnabled() != null) {
            setting.setPushEnabled(form.getPushEnabled() ? 1 : 0);
        }
        if (form.getDndEnabled() != null) {
            setting.setDndEnabled(form.getDndEnabled() ? 1 : 0);
        }
        if (CharSequenceUtil.isNotBlank(form.getDndStart())) {
            setting.setDndStart(LocalTime.parse(form.getDndStart(), TIME_FORMATTER));
        }
        if (CharSequenceUtil.isNotBlank(form.getDndEnd())) {
            setting.setDndEnd(LocalTime.parse(form.getDndEnd(), TIME_FORMATTER));
        }
        if (form.getPreferences() != null) {
            JSONObject prefs = CharSequenceUtil.isNotBlank(setting.getPreferences())
                    ? JSONUtil.parseObj(setting.getPreferences()) : new JSONObject();
            JSONObject formPrefs = new JSONObject(form.getPreferences());
            for (String key : formPrefs.keySet()) {
                Object formValue = formPrefs.get(key);
                Object existingValue = prefs.get(key);
                if (formValue instanceof JSONObject && existingValue instanceof JSONObject) {
                    JSONObject merged = new JSONObject();
                    merged.putAll((JSONObject) existingValue);
                    merged.putAll((JSONObject) formValue);
                    prefs.set(key, merged);
                } else {
                    prefs.set(key, formValue);
                }
            }
            setting.setPreferences(prefs.toString());
        }
        this.updateById(setting);
    }

    private SysNotificationSetting getOrCreateDefault(Long userId) {
        SysNotificationSetting setting = this.getOne(new LambdaQueryWrapper<SysNotificationSetting>()
                .eq(SysNotificationSetting::getUserId, userId));
        if (setting == null) {
            setting = new SysNotificationSetting();
            setting.setUserId(userId);
            setting.setPushEnabled(1);
            setting.setDndEnabled(0);
            setting.setDndStart(LocalTime.of(22, 0));
            setting.setDndEnd(LocalTime.of(8, 0));
            setting.setPreferences(DEFAULT_PREFERENCES);
            this.save(setting);
        }
        return setting;
    }

    private NotificationSettingsVO toVO(SysNotificationSetting setting) {
        NotificationSettingsVO vo = new NotificationSettingsVO();
        vo.setPushEnabled(setting.getPushEnabled() != null && setting.getPushEnabled() == 1);
        vo.setDndEnabled(setting.getDndEnabled() != null && setting.getDndEnabled() == 1);
        vo.setDndStart(setting.getDndStart() != null ? setting.getDndStart().format(TIME_FORMATTER) : null);
        vo.setDndEnd(setting.getDndEnd() != null ? setting.getDndEnd().format(TIME_FORMATTER) : null);
        if (CharSequenceUtil.isNotBlank(setting.getPreferences())) {
            vo.setPreferences(JSONUtil.parseObj(setting.getPreferences()).toBean(java.util.Map.class));
        }
        return vo;
    }
}
