import { AuthAPI, RegisterData } from "dehaze-sdk-js";
import defaultSettings from "@/settings";
import { LockOutlined, SafetyOutlined, UserOutlined } from "@ant-design/icons";
import { Button, Card, Form, Input, Tag } from "antd";
import React, { useCallback, useEffect, useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import "./index.scss";

export default function Register() {
  const [form] = Form.useForm();
  const [captchaBase64, setCaptchaBase64] = useState("");
  const [captchaKey, setCaptchaKey] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const getCaptcha = useCallback(() => {
    AuthAPI.getCaptcha().then((data) => {
      setCaptchaBase64(data.captchaBase64);
      setCaptchaKey(data.captchaKey);
    });
  }, []);

  useEffect(() => {
    getCaptcha();
  }, []);

  const handleRegister = (values: {
    username: string;
    password: string;
    confirmPassword: string;
    nickname: string;
    captchaCode: string;
  }) => {
    if (values.password !== values.confirmPassword) {
      form.setFields([
        { name: "confirmPassword", errors: ["两次密码输入不一致"] },
      ]);
      return;
    }

    setLoading(true);
    const data: RegisterData = {
      username: values.username,
      password: values.password,
      nickname: values.nickname,
      captchaKey,
      captchaCode: values.captchaCode,
    };

    AuthAPI.register(data)
      .then(() => {
        navigate("/login", { replace: true });
      })
      .catch(() => {
        form.setFieldsValue({ captchaCode: "" });
        getCaptcha();
      })
      .finally(() => {
        setLoading(false);
      });
  };

  return (
    <div className="login-container">
      <Card className="!border-none !bg-transparent !rounded-4% w-100 <sm:w-85">
        <div className="text-center relative">
          <h2>{defaultSettings.title}</h2>
          <Tag className="ml-2 absolute-rt">{defaultSettings.version}</Tag>
        </div>

        <Form form={form} className="login-form" onFinish={handleRegister}>
          <Form.Item
            name="username"
            rules={[
              { required: true, message: "请输入用户名" },
              {
                pattern: /^[a-zA-Z0-9_]{3,32}$/,
                message: "3-32位字母、数字、下划线",
              },
            ]}
          >
            <Input
              prefix={<UserOutlined />}
              placeholder="用户名"
              size="large"
              autoFocus
            />
          </Form.Item>

          <Form.Item
            name="nickname"
            rules={[{ required: true, message: "请输入昵称" }]}
          >
            <Input prefix={<UserOutlined />} placeholder="昵称" size="large" />
          </Form.Item>

          <Form.Item
            name="password"
            rules={[
              { required: true, message: "请输入密码" },
              {
                pattern: /^(?=.*[a-zA-Z])(?=.*\d).{6,20}$/,
                message: "6-20位，含字母和数字",
              },
            ]}
          >
            <Input.Password
              prefix={<LockOutlined />}
              placeholder="密码（6-20位，含字母和数字）"
              size="large"
            />
          </Form.Item>

          <Form.Item
            name="confirmPassword"
            rules={[{ required: true, message: "请确认密码" }]}
          >
            <Input.Password
              prefix={<LockOutlined />}
              placeholder="确认密码"
              size="large"
            />
          </Form.Item>

          <Form.Item
            name="captchaCode"
            rules={[{ required: true, message: "请输入验证码" }]}
          >
            <Input
              prefix={<SafetyOutlined />}
              placeholder="验证码"
              size="large"
              suffix={
                captchaBase64 ? (
                  <img
                    src={captchaBase64}
                    onClick={() => getCaptcha()}
                    alt="加载失败"
                    style={{ height: 34, cursor: "pointer" }}
                  />
                ) : (
                  <Button
                    type="link"
                    size="small"
                    onClick={getCaptcha}
                    style={{ height: 34, padding: 0 }}
                  >
                    加载验证码
                  </Button>
                )
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
              注册
            </Button>
          </Form.Item>
        </Form>

        <div className="text-center mt-4">
          <Link to="/login">已有账号？立即登录</Link>
        </div>
      </Card>
    </div>
  );
}
