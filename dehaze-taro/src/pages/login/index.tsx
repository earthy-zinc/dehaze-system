import React, { useState } from 'react';
import { View, Input, Button, Text, Form } from '@tarojs/components';
import { useDidShow } from '@tarojs/taro';
import './index.less';

const Login: React.FC = () => {
  const [isRegister, setIsRegister] = useState(false);
  const [formData, setFormData] = useState({
    username: '',
    password: '',
    confirmPassword: ''
  });
  const [loading, setLoading] = useState(false);

  useDidShow(() => {
    // 页面显示时的操作
  });

  const handleInput = (field: string, value: string) => {
    setFormData({
      ...formData,
      [field]: value
    });
  };

  const handleSubmit = () => {
    if (isRegister) {
      handleRegister();
    } else {
      handleLogin();
    }
  };

  const handleLogin = () => {
    if (!formData.username || !formData.password) {
      console.log('请输入用户名和密码');
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

  const handleRegister = () => {
    if (!formData.username || !formData.password || !formData.confirmPassword) {
      console.log('请填写所有字段');
      return;
    }
    
    if (formData.password !== formData.confirmPassword) {
      console.log('两次输入的密码不一致');
      return;
    }
    
    setLoading(true);
    // 模拟注册请求
    setTimeout(() => {
      console.log('注册:', { 
        username: formData.username, 
        password: formData.password 
      });
      setLoading(false);
      // 注册成功后自动登录或跳转到登录页
      setIsRegister(false);
    }, 800);
  };

  return (
    <View className='login-container'>
      <View className='login-card'>
        <View className='login-header'>
          <View className='app-logo'>雾</View>
          <Text className='app-title'>图像去雾系统</Text>
          <Text className='app-subtitle'>{isRegister ? '创建账户' : '登录账户'}</Text>
        </View>

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
          
          {isRegister && (
            <View className='form-group'>
              <Text className='form-label'>确认密码</Text>
              <Input
                className='form-input'
                placeholder='请再次输入密码'
                password
                value={formData.confirmPassword}
                onInput={(e) => handleInput('confirmPassword', e.detail.value)}
              />
            </View>
          )}
          
          <Button 
            className='form-button' 
            onClick={handleSubmit}
            disabled={loading}
          >
            {loading ? (isRegister ? '注册中...' : '登录中...') : (isRegister ? '注册' : '登录')}
          </Button>
          
          <View className='switch-container'>
            <Text className='switch-text'>
              {isRegister ? '已有账户?' : '没有账户?'}
              <Button 
                className='switch-button' 
                onClick={() => {
                  setIsRegister(!isRegister);
                  setFormData({
                    username: '',
                    password: '',
                    confirmPassword: ''
                  });
                }}
              >
                {isRegister ? '立即登录' : '立即注册'}
              </Button>
            </Text>
          </View>
        </Form>
      </View>

      <View className='login-footer'>
        <Text>© 2025 图像去雾系统</Text>
      </View>
    </View>
  );
};

export default Login;