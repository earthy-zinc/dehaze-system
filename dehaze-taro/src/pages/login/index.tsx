import React, { useEffect, useState } from 'react';
import { AuthAPI, CaptchaResult } from "dehaze-sdk-js";
import { View, Input, Button, Text, Form, Image } from '@tarojs/components';
import { useDidShow } from '@tarojs/taro';
import './index.less';

const Login: React.FC = () => {
  const [isRegister, setIsRegister] = useState(false);
  const [formData, setFormData] = useState({
    username: '',
    password: '',
    captcha: ''
  });
  const [loading, setLoading] = useState(false);
  const [captcha, setCaptcha] = useState<CaptchaResult>({
    captchaBase64: '',
    captchaKey: ''
  });

  const [captchaError, setCaptchaError] = useState('');

  useDidShow(async () => {

  });

  useEffect(() => {
    AuthAPI
      .getCaptcha()
      .then((res) => setCaptcha(res));
  }, []);
  const handleInput = (field: string, value: string) => {
    setFormData({
      ...formData,
      [field]: value
    });

    if (field === 'captcha') {
      setCaptchaError('');
    }
  };

  const handleSubmit = () => {
    if (!formData.username || !formData.password || !formData.captcha) {
      console.log('请输入所有字段');
      return;
    }

    if (parseInt(formData.captcha) !== 0) {
      setCaptchaError('验证码错误');
      return;
    }

    setLoading(true);
    // 模拟登录请求
    setTimeout(() => {
      console.log('登录:', {
        username: formData.username,
        password: formData.password
      });
      setLoading(false);
      // 登录成功后跳转到主页面
    }, 800);
  };


  const refreshCaptcha = async () => {
    const res = await AuthAPI.getCaptcha();
    setCaptcha(res);
    setFormData({
      ...formData,
      captcha: ''
    });
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
                value={formData.captcha}
                onInput={(e) => handleInput('captcha', e.detail.value)}
              />
              <Image className='captcha-image' src={captcha?.captchaBase64} onClick={refreshCaptcha}/>
            </View>
            {captchaError && <Text className='error-message'>{captchaError}</Text>}
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
