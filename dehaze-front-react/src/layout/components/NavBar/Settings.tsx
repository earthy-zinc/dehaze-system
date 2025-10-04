import { LayoutEnum } from "@/enums/LayoutEnum";
import { ThemeEnum } from "@/enums/ThemeEnum";
import { DisPatchType, RootState } from "@/store";
import { changeLanguage, changeSize } from "@/store/modules/appSlice";
import {
  changeLayout,
  changeTheme,
  changeThemeColor,
  toggleSidebarLogo,
  toggleTagsView,
  toggleWatermark,
} from "@/store/modules/settingsSlice";
import { MoonOutlined, SunOutlined } from "@ant-design/icons";
import { ColorPicker, Divider, Drawer, Select, Switch, Tooltip } from "antd";
import "./index.scss";
import { Color } from "antd/es/color-picker";
import React from "react";
import { useDispatch, useSelector } from "react-redux";

export const Settings: React.FC = () => {
  const settingsVisible = useSelector(
    (state: RootState) => state.settings.settingsVisible
  );

  const dispatch: DisPatchType = useDispatch();

  // 主题
  const theme = useSelector((state: RootState) => state.settings.theme);
  const isLightTheme = theme === ThemeEnum.LIGHT;
  const handleThemeChange = (checked: boolean) => {
    const theme = checked ? ThemeEnum.LIGHT : ThemeEnum.DARK;
    dispatch(changeTheme(theme));
  };

  // 布局
  const currentLayout = useSelector(
    (state: RootState) => state.settings.layout
  );
  const handleLayoutChange = (layout: LayoutEnum) => {
    dispatch(changeLayout(layout));
  };

  // 界面语言
  const languageOptions = [
    { value: "zh-CN", label: "简体中文" },
    { value: "en-US", label: "English" },
  ];
  const language = useSelector((state: RootState) => state.app.language);
  const defaultLanguage = languageOptions.find(
    (item) => item.value === language
  )?.label;
  const handleLanguageChange = (language: string) => {
    dispatch(changeLanguage(language));
  };

  // 字体大小
  const sizeOptions = [
    { value: "middle", label: "中号" },
    { value: "small", label: "小号" },
  ];
  const size = useSelector((state: RootState) => state.app.size);
  const defaultSize = sizeOptions.find((item) => item.value === size)?.label;
  const handleSizeChange = (size: string) => {
    dispatch(changeSize(size));
  };

  // 水印
  const watermarkEnabled = useSelector(
    (state: RootState) => state.settings.watermarkEnabled
  );
  const handleWatermarkChange = (checked: boolean) => {
    dispatch(toggleWatermark());
  };

  // 侧边栏图标
  const sidebarLogo = useSelector(
    (state: RootState) => state.settings.sidebarLogo
  );
  const handleSidebarLogoChange = (checked: boolean) => {
    dispatch(toggleSidebarLogo());
  };

  // 页面标签
  const tagsView = useSelector((state: RootState) => state.settings.tagsView);
  const handleTagsViewChange = (checked: boolean) => {
    dispatch(toggleTagsView());
  };

  // 主题颜色
  const themeColor = useSelector(
    (state: RootState) => state.settings.themeColor
  );

  const handleThemeColorChange = (color: Color) => {
    dispatch(changeThemeColor(color.toHex()));
  };
  return (
    <Drawer
      title="系统设置"
      onClose={() => dispatch({ type: "settings/toggleSettingsVisible" })}
      open={settingsVisible}
    >
      <Divider plain>主题设置</Divider>
      <div className="settings-option" style={{ justifyContent: "center" }}>
        <MoonOutlined className={`${!isLightTheme ? "active" : ""}`} />
        <Switch
          style={{ margin: "0 8px" }}
          checkedChildren="明亮"
          unCheckedChildren="暗黑"
          defaultChecked={isLightTheme}
          onChange={handleThemeChange}
        />
        <SunOutlined className={`${isLightTheme ? "active" : ""}`} />
      </div>

      <Divider plain>界面设置</Divider>

      <div className="settings-option">
        <span className="text-xs">主题颜色</span>
        <ColorPicker onChangeComplete={handleThemeColorChange} />
      </div>

      <div className="settings-option">
        <span className="text-xs">开启页面标签</span>
        <Switch defaultChecked={tagsView} onChange={handleTagsViewChange} />
      </div>

      <div className="settings-option">
        <span className="text-xs">固定页面标签</span>
        {/* TODO */}
        <Switch />
      </div>

      <div className="settings-option">
        <span className="text-xs">侧边栏图标</span>
        <Switch
          defaultChecked={sidebarLogo}
          onChange={handleSidebarLogoChange}
        />
      </div>

      <div className="settings-option">
        <span className="text-xs">开启水印</span>
        <Switch
          defaultChecked={watermarkEnabled}
          onChange={handleWatermarkChange}
        />
      </div>

      <div className="settings-option">
        <span className="text-xs">字体大小</span>
        <Select
          defaultValue={defaultSize}
          style={{ width: 100 }}
          options={sizeOptions}
          onChange={handleSizeChange}
        />
      </div>

      <div className="settings-option">
        <span className="text-xs">界面语言</span>
        <Select
          defaultValue={defaultLanguage}
          style={{ width: 100 }}
          onSelect={handleLanguageChange}
          options={languageOptions}
        />
      </div>
      <Divider plain>导航设置</Divider>

      <div className="settings-nav">
        <Tooltip title="左侧模式" placement="bottom">
          <div
            className={`layout-item left ${currentLayout === LayoutEnum.LEFT ? "active" : ""}`}
            onClick={() => handleLayoutChange(LayoutEnum.LEFT)}
          >
            <div></div>
            <div></div>
          </div>
        </Tooltip>

        <Tooltip title="顶部模式" placement="bottom">
          <div
            className={`layout-item top ${currentLayout === LayoutEnum.TOP ? "active" : ""}`}
            onClick={() => handleLayoutChange(LayoutEnum.TOP)}
          >
            <div></div>
            <div></div>
          </div>
        </Tooltip>

        <Tooltip title="混合模式" placement="bottom">
          <div
            className={`layout-item mix ${currentLayout === LayoutEnum.MIX ? "active" : ""}`}
            onClick={() => handleLayoutChange(LayoutEnum.MIX)}
          >
            <div></div>
            <div></div>
          </div>
        </Tooltip>
      </div>
    </Drawer>
  );
};
