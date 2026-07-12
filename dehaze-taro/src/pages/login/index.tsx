import React, { useEffect, useState } from 'react';
import {Button, Form, Image, Input, Text, View} from '@tarojs/components';
import Taro, { useDidShow } from '@tarojs/taro';
import './index.less';
import { AuthAPI, CaptchaResult } from "dehaze-sdk-js";
import { useGlobalContext } from '@/stores/global';

const Login: React.FC = () => {
  const { login } = useGlobalContext();
  const [formData, setFormData] = useState({
    username: 'admin',
    password: '123456',
    captchaCode: ''
  });
  const [loading, setLoading] = useState(false);
  const [captcha, setCaptcha] = useState<CaptchaResult>({
    captchaBase64: '',
    captchaKey: ''
  });

  useDidShow(async () => {

  });

  useEffect(() => {
    getCaptcha();
  }, []);

  const getCaptcha = async () => {
    try {
      const res = await AuthAPI.getCaptcha();
      setCaptcha(res);
      setFormData(prev => ({
        ...prev,
        captchaCode: ''
      }));
    } catch (error) {
      console.error('获取验证码失败:', error);
      Taro.showToast({ title: '获取验证码失败', icon: 'none' });
    }
  };

  const handleInput = (field: string, value: string) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleSubmit = async () => {
    // 表单验证
    const trimmedUsername = formData.username?.trim();
    const trimmedPassword = formData.password?.trim();
    const trimmedCaptcha = formData.captchaCode?.trim();

    if (!trimmedUsername) {
      Taro.showToast({ title: '请输入用户名', icon: 'none' });
      return;
    }

    if (!trimmedPassword) {
      Taro.showToast({ title: '请输入密码', icon: 'none' });
      return;
    }

    if (!trimmedCaptcha) {
      Taro.showToast({ title: '请输入验证码', icon: 'none' });
      return;
    }

    try {
      setLoading(true);

      const loginData = {
        username: trimmedUsername,
        password: trimmedPassword,
        captchaKey: captcha.captchaKey,
        captchaCode: trimmedCaptcha
      };

      // 使用全局 login 方法，会自动保存 token、用户信息、权限到 storage 和全局状态
      await login(loginData);

      // 登录成功
      Taro.showToast({ title: '登录成功', icon: 'success' });

      // 延迟跳转，让用户看到成功提示。登录后进入首页（tabbar 页面）
      setTimeout(() => {
        Taro.switchTab({ url: '/pages/home/index' });
      }, 1000);

    } catch (error: any) {
      // 打印完整错误对象，便于诊断
      console.error('登录失败:', error);
      const errMsg = error?.response?.data?.msg || error?.message || '登录失败，请检查用户名和密码';
      Taro.showToast({ title: errMsg, icon: 'none' });

      // 登录失败，刷新验证码
      await getCaptcha();
      setFormData(prev => ({ ...prev, captchaCode: '' }));
    } finally {
      setLoading(false);
    }
  };


  const refreshCaptcha = () => {
    getCaptcha();
  };

  return (
    <View className='login-container'>
      {/* 顶部 Logo 区 */}
      <View className='login-header'>
        <View className='logo-circle'>
          <Text style={{ fontSize: '56rpx', color: '#ffffff' }}>去雾</Text>
        </View>
        <Text className='app-title'>图像去雾系统</Text>
        <Text className='app-slogan'>专业级图像处理 · 深度学习算法</Text>
      </View>

      {/* 登录卡片 */}
      <View className='login-card'>
        <Form className='form-container'>
          <View className='form-group'>
            <Text className='form-label'>用户名</Text>
            <Input
              className='form-input'
              placeholder='请输入用户名'
              value={formData.username}
              onInput={(e) => handleInput('username', e.detail.value)}
            />
          </View>

          <View className='form-group'>
            <Text className='form-label'>密码</Text>
            <Input
              className='form-input'
              placeholder='请输入密码'
              password
              value={formData.password}
              onInput={(e) => handleInput('password', e.detail.value)}
            />
          </View>

          <View className='form-group'>
            <Text className='form-label'>验证码</Text>
            <View className='captcha-container'>
              <Input
                className='form-input'
                placeholder='请输入验证码'
                value={formData.captchaCode}
                onInput={(e) => handleInput('captchaCode', e.detail.value)}
              />
              <Image className='captcha-image' src={captcha?.captchaBase64} onClick={refreshCaptcha}/>
            </View>
          </View>

          <Button
            className='form-button'
            onClick={handleSubmit}
            disabled={loading}
          >
            {loading ? '登录中...' : '登 录'}
          </Button>

          <View className='footer-info'>
            <Text className='info-text'>账号 admin / 密码 123456</Text>
          </View>
        </Form>
      </View>

      {/* 底部版权 */}
      <View className='login-footer'>
        <View className='login-footer-text'>
          <Text>Copyright © 2022 - 2024 Peixin Wu All Rights Reserved.</Text>
        </View>
        <View className='login-footer-text'>
          <Text>渝ICP备2024111923号-2</Text>
        </View>
      </View>
    </View>
  );
};

export default Login;
