import {
  MailOutlined,
  AppstoreOutlined,
  SettingOutlined,
} from "@ant-design/icons";
import { Menu, MenuProps } from "antd";
import { Logo } from "../Logo";
import "./index.scss";
import { useSelector } from "react-redux";
import { RootState } from "@/store";
import { SidebarStatusEnum } from "@/enums/SidebarStatusEnum";
import { menuItems } from "@/router";
import { useNavigate } from "react-router-dom";

type MenuItem = Required<MenuProps>["items"][number];

export const SideMenu: React.FC = () => {
  const navigate = useNavigate();
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
