package com.pei.dehaze.controller;

import com.pei.dehaze.common.constant.SecurityConstants;
import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.model.dto.CaptchaResult;
import com.pei.dehaze.model.dto.LoginForm;
import com.pei.dehaze.model.dto.LoginResult;
import com.pei.dehaze.model.dto.RegisterForm;
import com.pei.dehaze.model.vo.UserInfoVO;
import com.pei.dehaze.plugin.ratelimit.annotation.RateLimit;
import com.pei.dehaze.service.AuthService;
import com.pei.dehaze.service.SysUserService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.servlet.http.HttpServletResponse;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpHeaders;
import org.springframework.http.ResponseCookie;
import org.springframework.web.bind.annotation.*;

@Tag(name = "01.认证中心")
@RestController
@RequestMapping("/api/v1/auth")
@RequiredArgsConstructor
public class AuthController {

    private final AuthService authService;
    private final SysUserService userService;

    private static final long SESSION_MAX_AGE = 604800L;

    @Operation(summary = "登录")
    @RateLimit(key = "rate_limit:login:", timeWindow = 60, maxRequests = 10,
            type = RateLimit.LimitType.IP, message = "登录尝试过于频繁，请60秒后再试")
    @PostMapping("/login")
    public Result<LoginResult> login(@RequestBody @Valid LoginForm form, HttpServletResponse response) {
        LoginResult loginResult = authService.login(form);
        boolean rememberMe = Boolean.TRUE.equals(form.getRememberMe());
        setSessionCookie(response, loginResult.getSessionId(), rememberMe);
        return Result.success(loginResult);
    }

    @Operation(summary = "注册")
    @RateLimit(key = "rate_limit:register:", timeWindow = 60, maxRequests = 10,
            type = RateLimit.LimitType.IP, message = "注册请求过于频繁，请60秒后再试")
    @PostMapping("/register")
    public Result<LoginResult> register(@RequestBody @Valid RegisterForm form, HttpServletResponse response) {
        LoginResult result = authService.register(form);
        setSessionCookie(response, result.getSessionId(), false);
        return Result.success(result);
    }

    @Operation(summary = "注销")
    @PostMapping("/logout")
    public Result<Void> logout(HttpServletResponse response) {
        authService.logout();
        clearSessionCookie(response);
        return Result.success();
    }

    @Operation(summary = "获取验证码")
    @GetMapping("/captcha")
    public Result<CaptchaResult> getCaptcha() {
        CaptchaResult captcha = authService.getCaptcha();
        return Result.success(captcha);
    }

    @Operation(summary = "获取当前用户信息")
    @GetMapping("/me")
    public Result<UserInfoVO> me() {
        return Result.success(userService.getCurrentUserInfo());
    }

    private void setSessionCookie(HttpServletResponse response, String sessionId, boolean rememberMe) {
        long maxAge = rememberMe ? SESSION_MAX_AGE : -1;
        ResponseCookie cookie = ResponseCookie.from(SecurityConstants.SESSION_COOKIE_NAME, sessionId)
                .httpOnly(true)
                .secure(true)
                .sameSite("Lax")
                .path("/api")
                .maxAge(maxAge)
                .build();
        response.addHeader(HttpHeaders.SET_COOKIE, cookie.toString());
    }

    private void clearSessionCookie(HttpServletResponse response) {
        ResponseCookie cookie = ResponseCookie.from(SecurityConstants.SESSION_COOKIE_NAME, "")
                .httpOnly(true)
                .secure(true)
                .sameSite("Lax")
                .path("/api")
                .maxAge(0)
                .build();
        response.addHeader(HttpHeaders.SET_COOKIE, cookie.toString());
    }
}
