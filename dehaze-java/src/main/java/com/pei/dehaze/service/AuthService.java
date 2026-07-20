package com.pei.dehaze.service;

import com.pei.dehaze.model.dto.CaptchaResult;
import com.pei.dehaze.model.dto.LoginForm;
import com.pei.dehaze.model.dto.LoginResult;
import com.pei.dehaze.model.dto.RefreshTokenForm;

import java.util.Map;

/**
 * 认证服务接口
 */
public interface AuthService {

    /**
     * 登录
     *
     * @param form 登录表单（用户名、密码、验证码）
     * @return 登录结果
     */
    LoginResult login(LoginForm form);

    /**
     * 登出
     */
    void logout();

    /**
     * 获取验证码
     *
     * @return 验证码
     */
    CaptchaResult getCaptcha();

    /**
     * 获取当前用户认证信息
     *
     * @return 用户信息（userId, username, roles, permissions）
     */
    Map<String, Object> getAuthInfo();

    /**
     * 刷新令牌
     *
     * @param form 包含 refreshToken 的表单
     * @return 新的登录结果（含新的 accessToken 和 refreshToken）
     */
    LoginResult refreshToken(RefreshTokenForm form);
}
