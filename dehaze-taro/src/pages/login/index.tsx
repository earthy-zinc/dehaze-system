import React, { useState } from 'react';
import { View, Input, Button, Text, Form } from '@tarojs/components';
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
  const [captchaValue, setCaptchaValue] = useState('7 x 0 = ?');
  const [captchaError, setCaptchaError] = useState('');

  useDidShow(() => {
    // 页面显示时的操作
  });

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

    // 验证验证码
    const expectedCaptcha = calculateExpectedCaptcha();
    if (parseInt(formData.captcha) !== expectedCaptcha) {
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

  const calculateExpectedCaptcha = (): number => {
    // 简单的验证码计算逻辑
    const parts = captchaValue.split(' ');
    const num1 = parseInt(parts[0]);
    const operator = parts[1];
    const num2 = parseInt(parts[2]);

    switch (operator) {
      case 'x': return num1 * num2;
      case '+': return num1 + num2;
      case '-': return num1 - num2;
      default: return 0;
    }
  };

  const refreshCaptcha = () => {
    // 生成新的验证码
    const num1 = Math.floor(Math.random() * 10);
    const num2 = Math.floor(Math.random() * 10);
    const operators = ['x', '+', '-'];
    const operator = operators[Math.floor(Math.random() * operators.length)];

    setCaptchaValue(`${num1} ${operator} ${num2} = ?`);
    setFormData({
      ...formData,
      captcha: ''
    });
    setCaptchaError('');
  };

  return (
    <View className='login-container'>
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
              icon='user'
            />
          </View>

          <View className='form-group'>
            <Input
              className='form-input'
              placeholder='请输入密码'
              password
              value={formData.password}
              onInput={(e) => handleInput('password', e.detail.value)}
              icon='lock'
            />
          </View>

          <View className='form-group'>
            <View className='captcha-container'>
              <Input
                className={`form-input ${captchaError ? 'error' : ''}`}
                placeholder='请输入验证码'
                value={formData.captcha}
                onInput={(e) => handleInput('captcha', e.detail.value)}
                icon='shield'
              />
              <View className='captcha-image' onClick={refreshCaptcha}>
                <Text className='captcha-text'>{captchaValue}</Text>
              </View>
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
            <Text className='info-text'>用户名: admin</Text>
            <Text className='info-text'>密码: 123456</Text>
          </View>
        </Form>
      </View>

      <View className='login-footer'>
        <Text>Copyright © 2022 - 2024 Peixin Wu All Rights Reserved. 武沛鑫 版权所有</Text>
        <Text>渝ICP备2024111923号-2</Text>
      </View>
    </View>
  );
};

export default Login;
