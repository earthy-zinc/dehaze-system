import React, { useState, useEffect, useCallback } from "react";
import { View, Text, ScrollView, Switch, Picker } from "@tarojs/components";
import Taro from "@tarojs/taro";
import { NotificationSettingAPI } from "dehaze-sdk-js";
import type {
  NotificationSettings,
  NotificationSettingsForm,
} from "dehaze-sdk-js";
import PageLayout from "@/layout";
import "./index.less";

const MODULE_SWITCHES = [
  {
    key: "system",
    label: "系统通知",
    desc: "接收系统公告和重要通知",
    icon: "🔔",
  },
  {
    key: "business",
    label: "业务通知",
    desc: "接收业务流程相关通知",
    icon: "📋",
  },
  {
    key: "member",
    label: "会员通知",
    desc: "接收会员权益和到期提醒",
    icon: "👑",
  },
  {
    key: "activity",
    label: "活动通知",
    desc: "接收优惠活动和促销信息",
    icon: "🎉",
  },
];

const TIME_OPTIONS: string[] = [];
for (let h = 0; h < 24; h++) {
  for (let m = 0; m < 60; m += 30) {
    TIME_OPTIONS.push(
      `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`
    );
  }
}

const NotifyPage: React.FC = () => {
  const [settings, setSettings] = useState<NotificationSettings | null>(null);
  const [loading, setLoading] = useState(false);
  const [dndStartPickerIdx, setDndStartPickerIdx] = useState<number[]>([44]); // 22:00
  const [dndEndPickerIdx, setDndEndPickerIdx] = useState<number[]>([16]); // 08:00

  const loadSettings = useCallback(async () => {
    setLoading(true);
    try {
      const data = await NotificationSettingAPI.get();
      setSettings(data);
      if (data.dndStart) {
        const idx = TIME_OPTIONS.indexOf(data.dndStart);
        if (idx >= 0) setDndStartPickerIdx([idx]);
      }
      if (data.dndEnd) {
        const idx = TIME_OPTIONS.indexOf(data.dndEnd);
        if (idx >= 0) setDndEndPickerIdx([idx]);
      }
    } catch {
      Taro.showToast({ title: "加载失败", icon: "none" });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadSettings();
  }, [loadSettings]);

  const doSave = useCallback(async (data: NotificationSettingsForm) => {
    try {
      await NotificationSettingAPI.update(data);
      Taro.showToast({ title: "已保存", icon: "success" });
    } catch {
      Taro.showToast({ title: "保存失败", icon: "none" });
    }
  }, []);

  const togglePushEnabled = useCallback(
    (val: boolean) => {
      setSettings((prev) => (prev ? { ...prev, pushEnabled: val } : prev));
      doSave({ pushEnabled: val });
    },
    [doSave]
  );

  const toggleModuleSwitch = useCallback(
    (module: string, val: boolean) => {
      setSettings((prev) => {
        if (!prev) return prev;
        return {
          ...prev,
          preferences: {
            ...prev.preferences,
            moduleSwitches: {
              ...prev.preferences.moduleSwitches,
              [module]: val,
            },
          },
        };
      });
      doSave({
        preferences: {
          moduleSwitches: { [module]: val },
        },
      });
    },
    [doSave]
  );

  const toggleDndEnabled = useCallback(
    (val: boolean) => {
      setSettings((prev) => (prev ? { ...prev, dndEnabled: val } : prev));
      doSave({ dndEnabled: val });
    },
    [doSave]
  );

  const handleDndStartChange = useCallback(
    (e: { detail: { value: string | number | number[] } }) => {
      const idx = Number(
        Array.isArray(e.detail.value) ? e.detail.value[0] : e.detail.value
      );
      setDndStartPickerIdx([idx]);
      const time = TIME_OPTIONS[idx];
      setSettings((prev) => (prev ? { ...prev, dndStart: time } : prev));
      doSave({ dndStart: time });
    },
    [doSave]
  );

  const handleDndEndChange = useCallback(
    (e: { detail: { value: string | number | number[] } }) => {
      const idx = Number(
        Array.isArray(e.detail.value) ? e.detail.value[0] : e.detail.value
      );
      setDndEndPickerIdx([idx]);
      const time = TIME_OPTIONS[idx];
      setSettings((prev) => (prev ? { ...prev, dndEnd: time } : prev));
      doSave({ dndEnd: time });
    },
    [doSave]
  );

  const getModuleSwitchVal = (module: string): boolean => {
    return settings?.preferences?.moduleSwitches?.[module] !== false;
  };

  if (loading && !settings) {
    return (
      <PageLayout level="L2" title="消息设置">
        <View className="notify-settings-page">
          <View className="loading-wrapper">
            <Text>加载中...</Text>
          </View>
        </View>
      </PageLayout>
    );
  }

  return (
    <PageLayout level="L2" title="消息设置">
      <View className="notify-settings-page">
        <ScrollView scrollY className="notify-scroll">
          {/* 推送通知总开关 */}
          <View className="settings-group">
            <Text className="group-title">通知开关</Text>
            <View className="settings-card">
              <View className="settings-item">
                <View className="settings-item-left">
                  <Text className="settings-icon">📢</Text>
                  <View className="settings-item-info">
                    <Text className="settings-title">推送通知</Text>
                    <Text className="settings-desc">
                      接收所有类型的推送通知
                    </Text>
                  </View>
                </View>
                <Switch
                  checked={settings?.pushEnabled ?? true}
                  color="#3b82f6"
                  onChange={(e) => togglePushEnabled(e.detail.value)}
                />
              </View>

              {MODULE_SWITCHES.map((mod) => (
                <React.Fragment key={mod.key}>
                  <View className="settings-divider" />
                  <View className="settings-item">
                    <View className="settings-item-left">
                      <Text className="settings-icon">{mod.icon}</Text>
                      <View className="settings-item-info">
                        <Text className="settings-title">{mod.label}</Text>
                        <Text className="settings-desc">{mod.desc}</Text>
                      </View>
                    </View>
                    <Switch
                      checked={getModuleSwitchVal(mod.key)}
                      color="#3b82f6"
                      onChange={(e) =>
                        toggleModuleSwitch(mod.key, e.detail.value)
                      }
                    />
                  </View>
                </React.Fragment>
              ))}
            </View>
          </View>

          {/* 免打扰设置 */}
          <View className="settings-group">
            <Text className="group-title">免打扰</Text>
            <View className="settings-card">
              <View className="settings-item">
                <View className="settings-item-left">
                  <Text className="settings-icon">🌙</Text>
                  <View className="settings-item-info">
                    <Text className="settings-title">免打扰模式</Text>
                    <Text className="settings-desc">
                      开启后在设定时段内不接收通知
                    </Text>
                  </View>
                </View>
                <Switch
                  checked={settings?.dndEnabled ?? false}
                  color="#3b82f6"
                  onChange={(e) => toggleDndEnabled(e.detail.value)}
                />
              </View>

              {settings?.dndEnabled && (
                <>
                  <View className="settings-divider" />
                  <View className="settings-item">
                    <View className="settings-item-left">
                      <Text className="settings-icon">🕐</Text>
                      <Text className="settings-title">开始时间</Text>
                    </View>
                    <Picker
                      mode="selector"
                      range={TIME_OPTIONS}
                      value={dndStartPickerIdx[0]}
                      onChange={handleDndStartChange}
                    >
                      <Text className="settings-value">
                        {settings?.dndStart || "22:00"}
                      </Text>
                    </Picker>
                  </View>
                  <View className="settings-divider" />
                  <View className="settings-item">
                    <View className="settings-item-left">
                      <Text className="settings-icon">🕖</Text>
                      <Text className="settings-title">结束时间</Text>
                    </View>
                    <Picker
                      mode="selector"
                      range={TIME_OPTIONS}
                      value={dndEndPickerIdx[0]}
                      onChange={handleDndEndChange}
                    >
                      <Text className="settings-value">
                        {settings?.dndEnd || "08:00"}
                      </Text>
                    </Picker>
                  </View>
                </>
              )}
            </View>
          </View>

          <View className="notify-footer">
            <Text className="footer-text">设置实时生效</Text>
          </View>
        </ScrollView>
      </View>
    </PageLayout>
  );
};

export default NotifyPage;
