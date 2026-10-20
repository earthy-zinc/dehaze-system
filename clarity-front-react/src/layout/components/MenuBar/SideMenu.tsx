import { routesToMenuItems } from "@/router";
import { RootState } from "@/store";
import { HomeOutlined } from "@ant-design/icons";
import { Menu } from "antd";
import React, { useMemo } from "react";
import { useSelector } from "react-redux";
import { useNavigate } from "react-router-dom";
import { Logo } from "../Logo";
import "./index.scss";

export const SideMenu: React.FC = () => {
  const navigate = useNavigate();
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

  const handleMenuSelect = ({ key }: { key: string }) => {
    navigate(key);
  };

  return (
    <>
      <Logo />
      <Menu
        className="menu-container"
        mode="inline"
        items={menuItems}
        onSelect={handleMenuSelect}
      />
    </>
  );
};
