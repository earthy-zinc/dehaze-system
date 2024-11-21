import AuthAPI from "@/api/auth";
import { LoginData } from "@/api/auth/model";
import { ThemeEnum } from "@/enums/ThemeEnum";
import { DisPatchType } from "@/store";
import { login } from "@/store/modules/userSlice";
import {
  LockOutlined,
  MoonOutlined,
  SafetyOutlined,
  SunOutlined,
  UserOutlined,
} from "@ant-design/icons";
import { Button, Card, Form, Image, Input, Switch, Tooltip } from "antd";
import React, { useCallback, useEffect, useState } from "react";
import { useDispatch } from "react-redux";
import { useLocation, useNavigate } from "react-router-dom";

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
      <Card className="login-card">
        <div className="text-center relative">
          <h2>登录</h2>
          <span className="version-tag">v1.0</span>
        </div>
        <Form>
          <Form.Item>
            <Input
              prefix={<UserOutlined />}
              placeholder="用户名"
              value={loginData.username}
              onChange={(e) =>
                setLoginData({ ...loginData, username: e.target.value })
              }
            />
          </Form.Item>
          <Form.Item>
            <Tooltip title="大写锁定已开启" visible={isCapslock}>
              <Input
                prefix={<LockOutlined />}
                type="password"
                placeholder="密码"
                value={loginData.password}
                onChange={(e) =>
                  setLoginData({ ...loginData, password: e.target.value })
                }
                onKeyUp={(e) => setIsCapslock(e.getModifierState("CapsLock"))}
              />
            </Tooltip>
          </Form.Item>
          <Form.Item>
            <Input
              prefix={<SafetyOutlined />}
              placeholder="验证码"
              value={loginData.captchaCode}
              onChange={(e) =>
                setLoginData({ ...loginData, captchaCode: e.target.value })
              }
            />
            <Image
              src={captchaBase64}
              onClick={() => {
                getCaptcha();
              }}
            />
          </Form.Item>
          <Form.Item>
            <Button type="primary" loading={loading} onClick={handleLogin}>
              登录
            </Button>
          </Form.Item>
        </Form>
      </Card>
    </div>
  );
}
