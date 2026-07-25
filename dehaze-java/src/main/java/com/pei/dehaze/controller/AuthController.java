package com.pei.dehaze.controller;

import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.model.dto.CaptchaResult;
import com.pei.dehaze.model.dto.LoginForm;
import com.pei.dehaze.model.dto.LoginResult;
import com.pei.dehaze.model.dto.RefreshTokenForm;
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

    private static final String REFRESH_TOKEN_COOKIE = "refreshToken";
    private static final String REMEMBER_ME_COOKIE = "rememberMe";
    private static final long REFRESH_TOKEN_MAX_AGE = 604800L;

    @Operation(summary = "登录")
    @RateLimit(key = "rate_limit:login:", timeWindow = 60, maxRequests = 10,
            type = RateLimit.LimitType.IP, message = "登录尝试过于频繁，请60秒后再试")
    @PostMapping("/login")
    public Result<LoginResult> login(@RequestBody LoginForm form, HttpServletResponse response) {
        LoginResult loginResult = authService.login(form);
        boolean rememberMe = Boolean.TRUE.equals(form.getRememberMe());
        setRefreshTokenCookies(response, loginResult.getRefreshToken(), rememberMe);
        return Result.success(loginResult);
    }

    @Operation(summary = "注销")
    @PostMapping("/logout")
    public Result<Void> logout(HttpServletResponse response) {
        authService.logout();
        clearRefreshTokenCookies(response);
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

    @Operation(summary = "刷新令牌")
    @PostMapping("/refresh")
    public Result<LoginResult> refresh(
            @CookieValue(value = REFRESH_TOKEN_COOKIE, required = false) String cookieRefreshToken,
            @CookieValue(value = REMEMBER_ME_COOKIE, required = false) String rememberMeStr,
            @Valid @RequestBody(required = false) RefreshTokenForm form,
            HttpServletResponse response) {
        String refreshToken = cookieRefreshToken != null ? cookieRefreshToken
                : (form != null ? form.getRefreshToken() : null);
        LoginResult loginResult = authService.refreshToken(refreshToken);
        boolean rememberMe = "true".equals(rememberMeStr);
        setRefreshTokenCookies(response, loginResult.getRefreshToken(), rememberMe);
        return Result.success(loginResult);
    }

    private void setRefreshTokenCookies(HttpServletResponse response, String refreshToken, boolean rememberMe) {
        long maxAge = rememberMe ? REFRESH_TOKEN_MAX_AGE : -1;
        ResponseCookie refreshCookie = ResponseCookie.from(REFRESH_TOKEN_COOKIE, refreshToken)
                .httpOnly(true)
                .sameSite("Lax")
                .path("/")
                .maxAge(maxAge)
                .build();
        ResponseCookie rememberCookie = ResponseCookie.from(REMEMBER_ME_COOKIE, String.valueOf(rememberMe))
                .sameSite("Lax")
                .path("/")
                .maxAge(maxAge)
                .build();
        response.addHeader(HttpHeaders.SET_COOKIE, refreshCookie.toString());
        response.addHeader(HttpHeaders.SET_COOKIE, rememberCookie.toString());
    }

    private void clearRefreshTokenCookies(HttpServletResponse response) {
        ResponseCookie refreshCookie = ResponseCookie.from(REFRESH_TOKEN_COOKIE, "")
                .httpOnly(true)
                .sameSite("Lax")
                .path("/")
                .maxAge(0)
                .build();
        ResponseCookie rememberCookie = ResponseCookie.from(REMEMBER_ME_COOKIE, "")
                .sameSite("Lax")
                .path("/")
                .maxAge(0)
                .build();
        response.addHeader(HttpHeaders.SET_COOKIE, refreshCookie.toString());
        response.addHeader(HttpHeaders.SET_COOKIE, rememberCookie.toString());
    }
}
