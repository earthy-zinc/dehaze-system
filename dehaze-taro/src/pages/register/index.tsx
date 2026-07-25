import React, { useEffect, useState } from "react";
import { Button, Image, Input, Text, View } from "@tarojs/components";
import Taro from "@tarojs/taro";
import { AuthAPI, CaptchaResult, RegisterData } from "dehaze-sdk-js";
import { getErrorMessage } from "@/utils/error";
import "./index.less";

const Register: React.FC = () => {
  const [formData, setFormData] = useState({
    username: "",
    password: "",
    confirmPassword: "",
    nickname: "",
    captchaCode: "",
  });
  const [loading, setLoading] = useState(false);
  const [captcha, setCaptcha] = useState<CaptchaResult>({
    captchaBase64: "",
    captchaKey: "",
  });

  useEffect(() => {
    getCaptcha();
  }, []);

  const getCaptcha = async () => {
    try {
      const res = await AuthAPI.getCaptcha();
      setCaptcha(res);
      setFormData((prev) => ({ ...prev, captchaCode: "" }));
    } catch (error) {
      Taro.showToast({ title: "获取验证码失败", icon: "none" });
    }
  };

  const handleSubmit = async () => {
    const u = formData.username.trim();
    const p = formData.password.trim();
    const cp = formData.confirmPassword.trim();
    const n = formData.nickname.trim();
    const cc = formData.captchaCode.trim();

    if (!u) {
      Taro.showToast({ title: "请输入用户名", icon: "none" });
      return;
    }
    if (!n) {
      Taro.showToast({ title: "请输入昵称", icon: "none" });
      return;
    }
    if (!p) {
      Taro.showToast({ title: "请输入密码", icon: "none" });
      return;
    }
    if (p !== cp) {
      Taro.showToast({ title: "两次密码不一致", icon: "none" });
      return;
    }
    if (!cc) {
      Taro.showToast({ title: "请输入验证码", icon: "none" });
      return;
    }

    setLoading(true);
    const data: RegisterData = {
      username: u,
      password: p,
      nickname: n,
      captchaKey: captcha.captchaKey,
      captchaCode: cc,
    };
    try {
      await AuthAPI.register(data);
      Taro.showToast({ title: "注册成功", icon: "success" });
      setTimeout(() => {
        Taro.reLaunch({ url: "/pages/login/index" });
      }, 1000);
    } catch (e: unknown) {
      const err = e as { response?: { data?: { msg?: string } } };
      Taro.showToast({
        title: err?.response?.data?.msg || getErrorMessage(e, "注册失败"),
        icon: "none",
      });
      await getCaptcha();
    } finally {
      setLoading(false);
    }
  };

  return (
    <View className="login-container">
      <View className="login-header">
        <View className="logo-circle">
          <Text style={{ fontSize: "56rpx", color: "#ffffff" }}>注册</Text>
        </View>
        <Text className="app-title">图像去雾系统</Text>
        <Text className="app-desc">创建新账号</Text>
      </View>
      <View className="login-form">
        <Input
          className="input-field"
          placeholder="用户名（3-32位字母数字下划线）"
          value={formData.username}
          onInput={(e) =>
            setFormData((prev) => ({ ...prev, username: e.detail.value }))
          }
        />
        <Input
          className="input-field"
          placeholder="昵称"
          value={formData.nickname}
          onInput={(e) =>
            setFormData((prev) => ({ ...prev, nickname: e.detail.value }))
          }
        />
        <Input
          className="input-field"
          password
          placeholder="密码（6-20位含字母和数字）"
          value={formData.password}
          onInput={(e) =>
            setFormData((prev) => ({ ...prev, password: e.detail.value }))
          }
        />
        <Input
          className="input-field"
          password
          placeholder="确认密码"
          value={formData.confirmPassword}
          onInput={(e) =>
            setFormData((prev) => ({
              ...prev,
              confirmPassword: e.detail.value,
            }))
          }
        />
        <View className="captcha-row">
          <Input
            className="captcha-input"
            placeholder="验证码"
            value={formData.captchaCode}
            onInput={(e) =>
              setFormData((prev) => ({ ...prev, captchaCode: e.detail.value }))
            }
          />
          <Image
            className="captcha-img"
            src={captcha.captchaBase64}
            onClick={getCaptcha}
          />
        </View>
        <Button
          className="login-btn"
          loading={loading}
          disabled={loading}
          onClick={handleSubmit}
        >
          注 册
        </Button>
        <View
          className="register-link"
          onClick={() => Taro.reLaunch({ url: "/pages/login/index" })}
        >
          <Text>已有账号？立即登录</Text>
        </View>
      </View>
    </View>
  );
};

export default Register;
