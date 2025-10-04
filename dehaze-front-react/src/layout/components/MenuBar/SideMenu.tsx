import { menuItems } from "@/router";
import { Menu } from "antd";
import React from "react";
import { useNavigate } from "react-router-dom";
import { Logo } from "../Logo";
import "./index.scss";

export const SideMenu: React.FC = () => {
  const navigate = useNavigate();
  const handleMenuSelect = ({ item }: { item: any }) => {
    navigate(item.props.path);
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
