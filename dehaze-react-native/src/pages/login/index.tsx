/**
 * 登录页
 *
 * 对接 dehaze-sdk-js AuthAPI：
 * - 登录（用户名 + 密码 + 图形验证码）
 * - 获取验证码图片
 * - 登录成功后由 AuthContext 跳转 Home
 */
import { useAuth } from '@/store';
import type { RootStackParamList } from '@/routes/types';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import { AuthAPI } from 'dehaze-sdk-js';
import type { CaptchaResult } from 'dehaze-sdk-js';
import React, { useEffect, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Image,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

type LoginScreenProps = NativeStackScreenProps<RootStackParamList, 'Login'>;

const LoginScreen: React.FC<LoginScreenProps> = () => {
  const { login } = useAuth();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [captchaCode, setCaptchaCode] = useState('');
  const [captcha, setCaptcha] = useState<CaptchaResult | null>(null);
  const [captchaLoading, setCaptchaLoading] = useState(false);
  const [loading, setLoading] = useState(false);

  const loadCaptcha = async () => {
    setCaptchaLoading(true);
    try {
      const result = await AuthAPI.getCaptcha();
      // 确保 Base64 带 data URI 前缀
      const base64 = result.captchaBase64.startsWith('data:')
        ? result.captchaBase64
        : `data:image/png;base64,${result.captchaBase64}`;
      setCaptcha({ ...result, captchaBase64: base64 });
    } catch (e) {
      setCaptcha(null);
    } finally {
      setCaptchaLoading(false);
    }
  };

  useEffect(() => {
    loadCaptcha();
  }, []);

  const handleLogin = async () => {
    if (!username || !password) {
      Alert.alert('提示', '请输入用户名和密码');
      return;
    }
    if (captcha && !captchaCode) {
      Alert.alert('提示', '请输入验证码');
      return;
    }

    setLoading(true);
    try {
      await login({
        username,
        password,
        captchaKey: captcha?.captchaKey,
        captchaCode: captcha ? captchaCode : undefined,
      });
      // 登录成功后 AuthContext 更新 isAuthenticated，路由守卫自动跳转 Home
    } catch (e: any) {
      Alert.alert('登录失败', e?.message || '用户名或密码错误');
      // 重新加载验证码
      loadCaptcha();
      setCaptchaCode('');
    } finally {
      setLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContainer}>
        <View style={styles.card}>
          <View style={styles.header}>
            <View style={styles.logoContainer}>
              <Text style={styles.logo}>雾</Text>
            </View>
            <Text style={styles.title}>图像去雾系统</Text>
            <Text style={styles.subtitle}>登录账户</Text>
          </View>

          <View style={styles.form}>
            <View style={styles.inputGroup}>
              <TextInput
                style={styles.input}
                placeholder="请输入用户名"
                value={username}
                onChangeText={setUsername}
                autoCapitalize="none"
                returnKeyType="next"
              />
            </View>

            <View style={styles.inputGroup}>
              <TextInput
                style={styles.input}
                placeholder="请输入密码"
                value={password}
                onChangeText={setPassword}
                secureTextEntry
                returnKeyType={captcha ? 'next' : 'done'}
              />
            </View>

            {captcha && (
              <View style={styles.captcha}>
                <TextInput
                  style={[styles.input, styles.captchaInput]}
                  placeholder="请输入验证码"
                  value={captchaCode}
                  onChangeText={setCaptchaCode}
                  autoCapitalize="none"
                  returnKeyType="done"
                />
                <TouchableOpacity
                  style={styles.captchaContainer}
                  onPress={loadCaptcha}
                  disabled={captchaLoading}
                >
                  {captchaLoading ? (
                    <ActivityIndicator size="small" />
                  ) : (
                    <Image
                      style={styles.captchaImage}
                      source={{ uri: captcha.captchaBase64 }}
                      resizeMode="contain"
                    />
                  )}
                </TouchableOpacity>
              </View>
            )}

            <TouchableOpacity
              style={styles.button}
              onPress={handleLogin}
              disabled={loading}
            >
              {loading ? (
                <ActivityIndicator color="white" />
              ) : (
                <Text style={styles.buttonText}>登录</Text>
              )}
            </TouchableOpacity>
          </View>
        </View>

        <View style={styles.footer}>
          <Text style={styles.footerText}>
            Copyright © 2022 - 2024 DehazeSystem All Rights Reserved.
          </Text>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f7fa',
  },
  scrollContainer: {
    flexGrow: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: 20,
  },
  card: {
    width: '90%',
    maxWidth: 400,
    backgroundColor: 'white',
    borderRadius: 16,
    padding: 24,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 3.84,
    elevation: 5,
  },
  header: {
    alignItems: 'center',
    marginBottom: 24,
  },
  logoContainer: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: '#667eea',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 16,
  },
  logo: {
    fontSize: 30,
    fontWeight: 'bold',
    color: 'white',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    color: '#666',
  },
  form: {
    width: '100%',
  },
  inputGroup: {
    marginBottom: 16,
  },
  captcha: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 16,
  },
  captchaInput: {
    flex: 1,
  },
  captchaContainer: {
    justifyContent: 'center',
    alignItems: 'center',
    width: 120,
    height: 50,
  },
  captchaImage: {
    width: 120,
    height: 50,
  },
  input: {
    height: 50,
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    paddingHorizontal: 12,
    fontSize: 16,
    backgroundColor: '#fafafa',
  },
  button: {
    height: 50,
    backgroundColor: '#667eea',
    borderRadius: 8,
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: 8,
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  footer: {
    marginTop: 24,
    alignItems: 'center',
  },
  footerText: {
    marginTop: 6,
    fontSize: 12,
    color: '#999',
  },
});

export default LoginScreen;
