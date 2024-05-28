import {
  Breadcrumb,
  ColorPicker,
  Divider,
  Drawer,
  Select,
  Switch,
  Tooltip,
} from "antd";
import "./index.scss";
import { useDispatch, useSelector } from "react-redux";
import { RootState } from "@/store";

export const Settings: React.FC = () => {
  const settingsVisiable = useSelector(
    (state: RootState) => state.settings.settingsVisiable
  );
  const dispatch = useDispatch();
  return (
    <Drawer
      title="系统设置"
      onClose={() => dispatch({ type: "settings/toggleSettingsVisiable" })}
      open={settingsVisiable}
    >
      <Divider plain>主题设置</Divider>
      <div className="settings-option" style={{ justifyContent: "center" }}>
        <Switch
          checkedChildren="明亮"
          unCheckedChildren="暗黑"
          defaultChecked
        />
      </div>

      <Divider plain>界面设置</Divider>

      <div className="settings-option">
        <span className="text-xs">主题颜色</span>
        <ColorPicker defaultValue="#1677ff" />
      </div>

      <div className="settings-option">
        <span className="text-xs">开启页面标签</span>
        <Switch />
      </div>

      <div className="settings-option">
        <span className="text-xs">固定页面标签</span>
        <Switch />
      </div>

      <div className="settings-option">
        <span className="text-xs">侧边栏图标</span>
        <Switch />
      </div>

      <div className="settings-option">
        <span className="text-xs">开启水印</span>
        <Switch />
      </div>

      <div className="settings-option">
        <span className="text-xs">字体大小</span>
        <Select
          defaultValue="sample"
          style={{ width: 100 }}
          options={[{ value: "sample", label: <span>默认</span> }]}
        />
      </div>

      <div className="settings-option">
        <span className="text-xs">界面语言</span>
        <Select
          defaultValue="sample"
          style={{ width: 100 }}
          options={[{ value: "sample", label: <span>中文</span> }]}
        />
      </div>
      <Divider plain>导航设置</Divider>

      <div className="settings-nav">
        <Tooltip title="左侧模式" placement="bottom">
          <div className="layout-item left">
            <div></div>
            <div></div>
          </div>
        </Tooltip>

        <Tooltip title="顶部模式" placement="bottom">
          <div className="layout-item top">
            <div></div>
            <div></div>
          </div>
        </Tooltip>

        <Tooltip title="混合模式" placement="bottom">
          <div className="layout-item mix">
            <div></div>
            <div></div>
          </div>
        </Tooltip>
      </div>
    </Drawer>
  );
};
