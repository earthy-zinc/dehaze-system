import {
  NotificationSettingAPI,
  type NotificationSettings,
  type NotificationSettingsForm,
} from "dehaze-sdk-js";
import {
  ArrowLeftOutlined,
  BellOutlined,
  CheckOutlined,
} from "@ant-design/icons";
import { Button, Spin, Switch, TimePicker, message } from "antd";
import dayjs from "dayjs";
import React, { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import "./settings.scss";

const TYPE_CHANNEL_LIST = [
  {
    key: "announcement",
    label: "系统公告",
    desc: "系统维护、功能更新、活动通知等",
  },
  {
    key: "business",
    label: "业务通知",
    desc: "订单状态变更、任务完成、反馈回复等",
  },
  {
    key: "member",
    label: "会员通知",
    desc: "等级变更、到期预警、权益更新等",
  },
];

const MODULE_SWITCH_LIST = [
  { key: "prediction", label: "去雾处理", desc: "去雾任务完成、失败等通知" },
  { key: "feedback", label: "反馈评价", desc: "反馈回复、评价处理等通知" },
  { key: "announcement", label: "系统公告", desc: "管理员发送的系统公告" },
];

interface SettingsState {
  pushEnabled: boolean;
  dndEnabled: boolean;
  dndStart: string;
  dndEnd: string;
  preferences: {
    typeChannels: Record<string, { push: boolean }>;
    moduleSwitches: Record<string, boolean>;
  };
}

const buildDefaultPreferences = () => ({
  typeChannels: {
    announcement: { push: true },
    business: { push: true },
    member: { push: true },
  },
  moduleSwitches: {
    prediction: true,
    feedback: true,
    announcement: true,
  },
});

const buildPayload = (state: SettingsState): NotificationSettingsForm => ({
  pushEnabled: state.pushEnabled,
  dndEnabled: state.dndEnabled,
  dndStart: state.dndStart,
  dndEnd: state.dndEnd,
  preferences: state.preferences,
});

const NotificationSettingsPage: React.FC = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [formData, setFormData] = useState<SettingsState | null>(null);
  const [originalSnapshot, setOriginalSnapshot] = useState("");

  const loadSettings = useCallback(() => {
    setLoading(true);
    NotificationSettingAPI.get()
      .then((data: NotificationSettings) => {
        const state: SettingsState = {
          pushEnabled: data.pushEnabled,
          dndEnabled: data.dndEnabled,
          dndStart: data.dndStart || "",
          dndEnd: data.dndEnd || "",
          preferences: data.preferences ?? buildDefaultPreferences(),
        };
        setFormData(state);
        setOriginalSnapshot(JSON.stringify(buildPayload(state)));
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    loadSettings();
  }, [loadSettings]);

  const hasChange = useMemo(() => {
    if (!formData) return false;
    return JSON.stringify(buildPayload(formData)) !== originalSnapshot;
  }, [formData, originalSnapshot]);

  const updateField = useCallback(
    <K extends keyof SettingsState>(key: K, value: SettingsState[K]) => {
      setFormData((prev) => (prev ? { ...prev, [key]: value } : prev));
    },
    []
  );

  const updateTypeChannel = useCallback((key: string, push: boolean) => {
    setFormData((prev) => {
      if (!prev) return prev;
      return {
        ...prev,
        preferences: {
          ...prev.preferences,
          typeChannels: {
            ...prev.preferences.typeChannels,
            [key]: { push },
          },
        },
      };
    });
  }, []);

  const updateModuleSwitch = useCallback((key: string, enabled: boolean) => {
    setFormData((prev) => {
      if (!prev) return prev;
      return {
        ...prev,
        preferences: {
          ...prev.preferences,
          moduleSwitches: {
            ...prev.preferences.moduleSwitches,
            [key]: enabled,
          },
        },
      };
    });
  }, []);

  const handleSave = useCallback(() => {
    if (!formData) return;
    setSaving(true);
    NotificationSettingAPI.update(buildPayload(formData))
      .then(() => {
        message.success("设置已保存");
        setOriginalSnapshot(JSON.stringify(buildPayload(formData)));
      })
      .catch((err) => message.error(err?.message || "保存失败"))
      .finally(() => setSaving(false));
  }, [formData]);

  return (
    <div className="app-container notification-settings">
      <div className="page-header">
        <div className="header-title">
          <Button
            type="link"
            icon={<ArrowLeftOutlined />}
            onClick={() => navigate("/message")}
          />
          <span className="title-text">通知设置</span>
        </div>
        <Button
          type="primary"
          disabled={!hasChange}
          loading={saving}
          icon={<CheckOutlined />}
          onClick={handleSave}
        >
          保存设置
        </Button>
      </div>

      <Spin spinning={loading}>
        {formData && (
          <>
            <section className="settings-section">
              <div className="section-header">
                <div className="section-title">
                  <BellOutlined className="section-icon" />
                  <span>推送通知</span>
                </div>
                <span className="section-desc">控制是否接收 APP 推送通知</span>
              </div>
              <div className="section-body">
                <div className="switch-row">
                  <div className="switch-label">
                    <span className="label-text">APP 推送</span>
                    <span className="label-desc">
                      开启后将通过 APP 推送接收消息通知
                    </span>
                  </div>
                  <Switch
                    checked={formData.pushEnabled}
                    onChange={(v) => updateField("pushEnabled", v)}
                  />
                </div>
              </div>
            </section>

            <section className="settings-section">
              <div className="section-header">
                <div className="section-title">
                  <BellOutlined className="section-icon" />
                  <span>免打扰</span>
                </div>
                <span className="section-desc">在指定时段内不接收推送通知</span>
              </div>
              <div className="section-body">
                <div className="switch-row">
                  <div className="switch-label">
                    <span className="label-text">开启免打扰</span>
                    <span className="label-desc">在指定时段内静默所有推送</span>
                  </div>
                  <Switch
                    checked={formData.dndEnabled}
                    onChange={(v) => updateField("dndEnabled", v)}
                  />
                </div>
                {formData.dndEnabled && (
                  <div className="dnd-time-row">
                    <div className="time-picker-item">
                      <label>开始时间</label>
                      <TimePicker
                        value={
                          formData.dndStart
                            ? dayjs(formData.dndStart, "HH:mm:ss")
                            : null
                        }
                        format="HH:mm"
                        placeholder="开始时间"
                        onChange={(_t, str) =>
                          updateField(
                            "dndStart",
                            typeof str === "string" ? str : ""
                          )
                        }
                      />
                    </div>
                    <div className="time-separator">至</div>
                    <div className="time-picker-item">
                      <label>结束时间</label>
                      <TimePicker
                        value={
                          formData.dndEnd
                            ? dayjs(formData.dndEnd, "HH:mm:ss")
                            : null
                        }
                        format="HH:mm"
                        placeholder="结束时间"
                        onChange={(_t, str) =>
                          updateField(
                            "dndEnd",
                            typeof str === "string" ? str : ""
                          )
                        }
                      />
                    </div>
                  </div>
                )}
              </div>
            </section>

            <section className="settings-section">
              <div className="section-header">
                <div className="section-title">
                  <BellOutlined className="section-icon" />
                  <span>按类型设置推送</span>
                </div>
                <span className="section-desc">
                  为不同类型的消息单独配置推送开关
                </span>
              </div>
              <div className="section-body">
                {TYPE_CHANNEL_LIST.map((item) => (
                  <div className="switch-row" key={item.key}>
                    <div className="switch-label">
                      <span
                        className={`label-text type-label type-${item.key}`}
                      >
                        <span className="type-dot" />
                        {item.label}
                      </span>
                      <span className="label-desc">{item.desc}</span>
                    </div>
                    <Switch
                      checked={
                        formData.preferences.typeChannels[item.key]?.push ??
                        true
                      }
                      onChange={(v) => updateTypeChannel(item.key, v)}
                    />
                  </div>
                ))}
              </div>
            </section>

            <section className="settings-section">
              <div className="section-header">
                <div className="section-title">
                  <BellOutlined className="section-icon" />
                  <span>模块通知</span>
                </div>
                <span className="section-desc">按业务模块开关对应的通知</span>
              </div>
              <div className="section-body">
                {MODULE_SWITCH_LIST.map((item) => (
                  <div className="switch-row" key={item.key}>
                    <div className="switch-label">
                      <span className="label-text">{item.label}</span>
                      <span className="label-desc">{item.desc}</span>
                    </div>
                    <Switch
                      checked={
                        formData.preferences.moduleSwitches[item.key] ?? true
                      }
                      onChange={(v) => updateModuleSwitch(item.key, v)}
                    />
                  </div>
                ))}
              </div>
            </section>
          </>
        )}
      </Spin>
    </div>
  );
};

export default NotificationSettingsPage;
