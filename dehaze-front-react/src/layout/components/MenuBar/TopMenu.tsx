import { routesToMenuItems } from "@/router";
import { Logo } from "@/layout/components/Logo";
import { Settings } from "@/layout/components/NavBar/Settings";
import { DisPatchType, RootState } from "@/store";
import { toggleSettingsVisible } from "@/store/modules/settingsSlice";
import { HomeOutlined, SettingOutlined } from "@ant-design/icons";
import { Menu } from "antd";
import React, { useMemo } from "react";
import { useDispatch, useSelector } from "react-redux";
import { useNavigate } from "react-router-dom";

export const TopMenu: React.FC = () => {
  const navigate = useNavigate();
  const dispatch: DisPatchType = useDispatch();
  const permissionRoutes = useSelector(
    (state: RootState) => state.permission.routes
  );

  const menuItems = useMemo(() => {
    const homeItem = {
      key: "/home",
      label: "首页",
      icon: <HomeOutlined />,
    };
    return [homeItem, ...routesToMenuItems(permissionRoutes)];
  }, [permissionRoutes]);

  const handleMenuSelect = ({ key }: { key: string }) => navigate(key);

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
