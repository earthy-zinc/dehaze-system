import React, { useEffect, useState } from 'react';
import {Button, Form, Image, Input, Text, View} from '@tarojs/components';
import Taro, { useDidShow } from '@tarojs/taro';
import {
  Toast,
} from '@taroify/core';
import './index.less';
import { AuthAPI, CaptchaResult } from "dehaze-sdk-js";

const Login: React.FC = () => {
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
      Toast.open({ message: '获取验证码失败', position: 'top' });
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
      Toast.open({ message: '请输入用户名', position: 'top' });
      return;
    }

    if (!trimmedPassword) {
      Toast.open({ message: '请输入密码', position: 'top' });
      return;
    }

    if (!trimmedCaptcha) {
      Toast.open({ message: '请输入验证码', position: 'top' });
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

      await AuthAPI.login(loginData);

      // 登录成功
      Toast.open({ message: '登录成功', position: 'top' });

      // 延迟跳转，让用户看到成功提示
      setTimeout(() => {
        Taro.redirectTo({url: '/pages/dashboard/index'});
      }, 1000);

    } catch (error) {
      console.error('登录失败:', error);
      Toast.open({ message: '登录失败，请检查用户名和密码', position: 'top' });

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
      <View className='login-header'>
      </View>

      <View className='login-card'>
        <View className='login-header'>
          <Text className='app-title'>图像去雾系统</Text>
          <Text className='version'>1.10.1</Text>
        </View>

        <Form className='form-container'>
          <View className='form-group'>
            <Input
              className='form-input'
              placeholder='请输入用户名'
              value={formData.username}
              onInput={(e) => handleInput('username', e.detail.value)}
            />
          </View>

          <View className='form-group'>
            <Input
              className='form-input'
              placeholder='请输入密码'
              password
              value={formData.password}
              onInput={(e) => handleInput('password', e.detail.value)}
            />
          </View>

          <View className='form-group'>
            <View className='captcha-container'>
              <Input
                className={`form-input`}
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
            <Text className='info-text mr-2'>用户名: admin</Text>
            <Text className='info-text'>密码: 123456</Text>
          </View>
        </Form>
      </View>

      <View className='login-footer'>
        <View className='login-footer-text'>
          <Text>Copyright © 2022 - 2024 Peixin Wu All Rights Reserved.</Text>
        </View>
        <View className='login-footer-text'>
          <Text>武沛鑫 版权所有</Text>
        </View>
        <View className='login-footer-text'>
          <Text>渝ICP备2024111923号-2</Text>
        </View>
      </View>
    </View>
  );
};

export default Login;
