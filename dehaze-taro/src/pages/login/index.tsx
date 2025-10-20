import React, { useEffect, useState } from 'react';
import { AuthAPI, CaptchaResult } from "dehaze-sdk-js";
import { View, Input, Button, Text, Form, Image } from '@tarojs/components';
import { useDidShow, useRouter } from '@tarojs/taro';
import './index.less';
import Taro from '@tarojs/taro';

const Login: React.FC = () => {
  const [isRegister, setIsRegister] = useState(false);
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

  const [captchaError, setCaptchaError] = useState('');
  const router = useRouter();

  useDidShow(async () => {

  });

  useEffect(() => {
    getCaptcha();
  }, []);

  const getCaptcha = () => {
    AuthAPI
      .getCaptcha()
      .then((res) => {
        setCaptcha(res);
        setFormData(prev => ({
          ...prev,
          captchaCode: ''
        }));
      });
  };

  const handleInput = (field: string, value: string) => {
    setFormData({
      ...formData,
      [field]: value
    });

    if (field === 'captchaCode') {
      setCaptchaError('');
    }
  };

  const handleSubmit = () => {
    if (!formData.username || !formData.password || !formData.captchaCode) {
      console.log('请输入所有字段');
      return;
    }

    setLoading(true);

    const loginData = {
      username: formData.username,
      password: formData.password,
      captchaKey: captcha.captchaKey,
      captchaCode: formData.captchaCode
    };

    AuthAPI
      .login(loginData)
      .then(() => {
        // 登录成功后跳转到主页面
        Taro.redirectTo({ url: '/pages/dashboard/index' }); // 假设有这样的页面
      })
      .catch(() => {
        // 登录失败，刷新验证码
        getCaptcha();
      })
      .finally(() => {
        setLoading(false);
      });
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
