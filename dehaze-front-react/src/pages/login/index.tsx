import AuthAPI from "@/api/auth";
import { LoginData } from "@/api/auth/model";
import { ThemeEnum } from "@/enums/ThemeEnum";
import defaultSettings from "@/settings";
import { DisPatchType } from "@/store";
import { login } from "@/store/modules/userSlice";
import {
  LockOutlined,
  MoonOutlined,
  SafetyOutlined,
  SunOutlined,
  UserOutlined,
} from "@ant-design/icons";
import { Button, Card, Form, Input, Switch, Tag, Tooltip } from "antd";
import React, { useCallback, useEffect, useState } from "react";
import { useDispatch } from "react-redux";
import { useLocation, useNavigate } from "react-router-dom";
import "./index.scss";

export default function Login() {
  const [loginData, setLoginData] = useState<LoginData>({
    username: "admin",
    password: "123456",
  });
  const [captchaBase64, setCaptchaBase64] = useState("");
  const [loading, setLoading] = useState(false);
  const [isCapslock, setIsCapslock] = useState(false);
  const [isDark, setIsDark] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();

  const dispatch: DisPatchType = useDispatch();

  const getCaptcha = useCallback(() => {
    AuthAPI.getCaptcha().then((data) => {
      setCaptchaBase64(data.captchaBase64);
      setLoginData({ ...loginData, captchaKey: data.captchaKey });
    });
  }, [loginData]);

  useEffect(() => {
    getCaptcha();
  }, []);

  const handleLogin = () => {
    setLoading(true);
    dispatch(login(loginData))
      .then(() => {
        const query = new URLSearchParams(location.search);
        const redirect = query.get("redirect") || "/";
        const otherQueryParams: Record<string, string> = {};
        query.forEach((value, key) => {
          if (key !== "redirect") {
            otherQueryParams[key] = value;
          }
        });
        navigate({
          pathname: redirect,
          search: new URLSearchParams(otherQueryParams).toString(),
        });
      })
      .catch(() => {
        getCaptcha();
      })
      .finally(() => {
        setLoading(false);
      });
  };

  const toggleTheme = () => {
    const newTheme = isDark ? ThemeEnum.LIGHT : ThemeEnum.DARK;
    setIsDark(!isDark);
    dispatch({ type: "settings/changeTheme", payload: newTheme });
  };

  return (
    <div className="login-container">
      <div className="absolute-lt flex-x-end p-3 w-full">
        <Switch
          checkedChildren={<MoonOutlined />}
          unCheckedChildren={<SunOutlined />}
          checked={isDark}
          onChange={toggleTheme}
        />
      </div>
      <Card className="!border-none !bg-transparent !rounded-4% w-100 <sm:w-85">
        <div className="text-center relative">
          <h2>{defaultSettings.title}</h2>
          <Tag className="ml-2 absolute-rt">{defaultSettings.version}</Tag>
        </div>

        <Form className="login-form">
          <Form.Item>
            <Input
              prefix={<UserOutlined />}
              placeholder="用户名"
              size="large"
              value={loginData.username}
              onChange={(e) =>
                setLoginData({ ...loginData, username: e.target.value })
              }
            />
          </Form.Item>
          <Form.Item>
            <Tooltip title="大写锁定已开启" open={isCapslock}>
              <Input
                prefix={<LockOutlined />}
                type="password"
                placeholder="密码"
                size="large"
                value={loginData.password}
                onChange={(e) =>
                  setLoginData({ ...loginData, password: e.target.value })
                }
                onKeyUp={(e) => setIsCapslock(e.getModifierState("CapsLock"))}
              />
            </Tooltip>
          </Form.Item>
          <Form.Item>
            <div className="flex-y-center w-full">
              <Input
                className="flex-1 absolute-lt"
                prefix={<SafetyOutlined />}
                placeholder="验证码"
                size="large"
                value={loginData.captchaCode}
                onChange={(e) =>
                  setLoginData({ ...loginData, captchaCode: e.target.value })
                }
              />
              <img
                className="rounded-tr-md rounded-br-md cursor-pointer relative h-[34px] top-1 left-55 z-36"
                src={captchaBase64}
                onClick={() => {
                  getCaptcha();
                }}
                alt="加载失败"
              />
            </div>
          </Form.Item>
          <Button
            className="w-full"
            size="large"
            type="primary"
            loading={loading}
            onClick={handleLogin}
          >
            登录
          </Button>
          <div className="mt-10 text-sm">
            <span>用户名: admin</span>
            <span className="ml-4"> 密码: 123456</span>
          </div>
        </Form>
      </Card>
      <div className="absolute bottom-1 text-[10px] text-center">
        <p>
          Copyright © 2022 - 2024 Peixin Wu All Rights Reserved. 武沛鑫
          版权所有
        </p>
        <p>渝ICP备2024111923号-2</p>
      </div>
    </div>
  );
}
