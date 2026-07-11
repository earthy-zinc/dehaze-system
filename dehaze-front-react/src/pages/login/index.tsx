import { AuthAPI, LoginData } from "dehaze-sdk-js";
import { ThemeEnum } from "@/enums/ThemeEnum";
import defaultSettings from "@/settings";
import { DisPatchType } from "@/store";
import { login } from "@/store/modules/userSlice";
import {
  EyeInvisibleOutlined,
  EyeTwoTone,
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
  const [form] = Form.useForm();
  const [captchaBase64, setCaptchaBase64] = useState("");
  const [captchaKey, setCaptchaKey] = useState("");
  const [loading, setLoading] = useState(false);
  const [isCapslock, setIsCapslock] = useState(false);
  const [isDark, setIsDark] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();

  const dispatch: DisPatchType = useDispatch();

  const getCaptcha = useCallback(() => {
    AuthAPI.getCaptcha().then((data) => {
      setCaptchaBase64(data.captchaBase64);
      setCaptchaKey(data.captchaKey);
    });
  }, []);

  useEffect(() => {
    getCaptcha();
  }, []);

  // 登录提交（Form 的 onFinish 自带回车提交支持）
  const handleLogin = (values: {
    username: string;
    password: string;
    captchaCode: string;
  }) => {
    setLoading(true);
    const loginData: LoginData = { ...values, captchaKey };
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
        // 登录失败：清空密码和验证码输入，并刷新验证码
        form.setFieldsValue({ password: "", captchaCode: "" });
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

        <Form
          form={form}
          className="login-form"
          initialValues={{ username: "admin", password: "123456" }}
          onFinish={handleLogin}
        >
          <Form.Item
            name="username"
            rules={[{ required: true, message: "请输入用户名" }]}
          >
            <Input
              prefix={<UserOutlined />}
              placeholder="用户名"
              size="large"
              autoFocus
            />
          </Form.Item>
          <Tooltip title="大写锁定已开启" open={isCapslock}>
            <Form.Item
              name="password"
              rules={[{ required: true, message: "请输入密码" }]}
            >
              <Input.Password
                prefix={<LockOutlined />}
                placeholder="密码"
                size="large"
                iconRender={(visible) =>
                  visible ? <EyeTwoTone /> : <EyeInvisibleOutlined />
                }
                onKeyUp={(e) =>
                  setIsCapslock(e.getModifierState("CapsLock"))
                }
              />
            </Form.Item>
          </Tooltip>
          <Form.Item
            name="captchaCode"
            rules={[{ required: true, message: "请输入验证码" }]}
          >
            <Input
              prefix={<SafetyOutlined />}
              placeholder="验证码"
              size="large"
              suffix={
                <img
                  src={captchaBase64}
                  onClick={() => getCaptcha()}
                  alt="加载失败"
                  style={{ height: 34, cursor: "pointer" }}
                />
              }
            />
          </Form.Item>
          <Form.Item>
            <Button
              className="w-full"
              size="large"
              type="primary"
              htmlType="submit"
              loading={loading}
            >
              登录
            </Button>
          </Form.Item>
        </Form>
        <div className="mt-10 text-sm">
          <span>用户名: admin</span>
          <span className="ml-4"> 密码: 123456</span>
        </div>
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
