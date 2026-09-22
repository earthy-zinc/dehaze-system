package com.pei.dehaze.service.impl;

import cn.hutool.core.text.CharSequenceUtil;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.pei.dehaze.mapper.SysNotificationSettingMapper;
import com.pei.dehaze.model.entity.SysNotificationSetting;
import com.pei.dehaze.model.form.NotificationSettingForm;
import com.pei.dehaze.model.vo.NotificationSettingsVO;
import com.pei.dehaze.security.util.SecurityUtils;
import com.pei.dehaze.service.NotificationSettingService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.util.Map;

@Slf4j
@Service
@RequiredArgsConstructor
public class NotificationSettingServiceImpl extends ServiceImpl<SysNotificationSettingMapper, SysNotificationSetting> implements NotificationSettingService {

    private static final DateTimeFormatter TIME_FORMATTER = DateTimeFormatter.ofPattern("HH:mm:ss");
    private static final String DEFAULT_PREFERENCES = "{\"typeChannels\":{\"announcement\":{\"push\":true},\"business\":{\"push\":false},\"member\":{\"push\":true}},\"moduleSwitches\":{\"prediction\":true,\"feedback\":true,\"announcement\":true}}";
    /** 默认偏好的解析结果：JSON 常量非法属编码错误，启动即暴露而非请求时兜底 */
    private static final Map<String, Object> DEFAULT_PREFERENCES_MAP = parseDefaultPreferences();

    private final ObjectMapper objectMapper;

    @Override
    public NotificationSettingsVO get() {
        Long userId = SecurityUtils.getUserId();
        // upsert 确保记录存在（复活软删行或新建）
        baseMapper.upsertByUser(userId, null, null, null);
        SysNotificationSetting setting = this.getOne(new LambdaQueryWrapper<SysNotificationSetting>()
                .eq(SysNotificationSetting::getUserId, userId));
        return toVO(setting != null ? setting : buildDefault(userId));
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void update(NotificationSettingForm form) {
        Long userId = SecurityUtils.getUserId();
        SysNotificationSetting setting = this.getOne(new LambdaQueryWrapper<SysNotificationSetting>()
                .eq(SysNotificationSetting::getUserId, userId));
        if (setting == null) {
            baseMapper.upsertByUser(userId, null, null, null);
            setting = this.getOne(new LambdaQueryWrapper<SysNotificationSetting>()
                    .eq(SysNotificationSetting::getUserId, userId));
        }
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

    private SysNotificationSetting buildDefault(Long userId) {
        SysNotificationSetting setting = new SysNotificationSetting();
        setting.setUserId(userId);
        setting.setPushEnabled(1);
        setting.setDndEnabled(0);
        setting.setDndStart(LocalTime.of(22, 0));
        setting.setDndEnd(LocalTime.of(8, 0));
        setting.setPreferences(DEFAULT_PREFERENCES);
        return setting;
    }

    private NotificationSettingsVO toVO(SysNotificationSetting setting) {
        NotificationSettingsVO vo = new NotificationSettingsVO();
        vo.setPushEnabled(setting.getPushEnabled() != null && setting.getPushEnabled() == 1);
        vo.setDndEnabled(setting.getDndEnabled() != null && setting.getDndEnabled() == 1);
        vo.setDndStart(setting.getDndStart() != null ? setting.getDndStart().format(TIME_FORMATTER) : null);
        vo.setDndEnd(setting.getDndEnd() != null ? setting.getDndEnd().format(TIME_FORMATTER) : null);
        // upsert 新建的行不写 preferences，回退默认值，避免细粒度偏好整块缺失
        if (CharSequenceUtil.isBlank(setting.getPreferences())) {
            vo.setPreferences(DEFAULT_PREFERENCES_MAP);
            return vo;
        }
        try {
            vo.setPreferences(objectMapper.readValue(setting.getPreferences(),
                    new TypeReference<Map<String, Object>>() {}));
        } catch (JsonProcessingException e) {
            log.warn("通知偏好 JSON 解析失败，回退默认值: {}", setting.getPreferences(), e);
            vo.setPreferences(DEFAULT_PREFERENCES_MAP);
        }
        return vo;
    }

    private static Map<String, Object> parseDefaultPreferences() {
        try {
            return new ObjectMapper().readValue(DEFAULT_PREFERENCES, new TypeReference<Map<String, Object>>() {});
        } catch (JsonProcessingException e) {
            throw new IllegalStateException("DEFAULT_PREFERENCES 不是合法 JSON", e);
        }
    }
}
