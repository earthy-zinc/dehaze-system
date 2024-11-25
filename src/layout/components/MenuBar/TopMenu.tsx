import { Logo } from "@/layout/components/Logo";
import { Settings } from "@/layout/components/NavBar/Settings";
import { menuItems } from "@/router";
import { DisPatchType } from "@/store";
import { toggleSettingsVisible } from "@/store/modules/settingsSlice";
import { SettingOutlined } from "@ant-design/icons";
import { Menu } from "antd";
import React from "react";
import { useDispatch } from "react-redux";
import { useNavigate } from "react-router-dom";

export const TopMenu: React.FC = () => {
  const navigate = useNavigate();
  const handleMenuSelect = ({ item }: { item: any }) =>
    navigate(item.props.path);

  const dispatch: DisPatchType = useDispatch();

  return (
    <>
      <div className="navbar-left">
        <Logo />
      </div>
      <Menu
        className="justify-center"
        style={{ flexGrow: 1 }}
        mode="horizontal"
        items={menuItems}
        onSelect={handleMenuSelect}
      />
      <div className="navbar-right">
        <button
          className="menu-status-icon"
          onClick={() => dispatch(toggleSettingsVisible())}
        >
          <SettingOutlined />
        </button>
      </div>
      <Settings />
    </>
  );
};
