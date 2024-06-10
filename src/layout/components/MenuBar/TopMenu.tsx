import { Logo } from "@/layout/components/Logo";
import { Settings } from "@/layout/components/NavBar/Settings";
import { menuItems } from "@/router";
import { SettingOutlined } from "@ant-design/icons";
import { Menu } from "antd";
import React from "react";
import { useDispatch } from "react-redux";
import { useNavigate } from "react-router-dom";

export const TopMenu: React.FC = () => {
  const navigate = useNavigate();
  const handleMenuSelect = ({ item }: { item: any }) => {
    navigate(item.props.path);
  };

  const dispatch = useDispatch();
  const handleSettingClick = () => {
    dispatch({
      type: "settings/toggleSettingsVisiable",
    });
  };

  return (
    <>
      <div className="navbar-left">
        <Logo />
      </div>
      <Menu
        style={{ flexGrow: 1 }}
        mode="horizontal"
        items={menuItems}
        onSelect={handleMenuSelect}
      />
      <div className="navbar-right">
        <div className="menu-status-icon" onClick={handleSettingClick}>
          <SettingOutlined />
        </div>
      </div>
      <Settings />
    </>
  );
};
