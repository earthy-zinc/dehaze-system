package com.pei.dehaze.controller;

import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.model.dto.CaptchaResult;
import com.pei.dehaze.model.dto.LoginForm;
import com.pei.dehaze.model.dto.LoginResult;
import com.pei.dehaze.model.dto.RefreshTokenForm;
import com.pei.dehaze.plugin.ratelimit.annotation.RateLimit;
import com.pei.dehaze.service.AuthService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@Tag(name = "01.认证中心")
@RestController
@RequestMapping("/api/v1/auth")
@RequiredArgsConstructor
public class AuthController {

    private final AuthService authService;

    @Operation(summary = "登录")
    @RateLimit(key = "rate_limit:login:", timeWindow = 60, maxRequests = 10,
            type = RateLimit.LimitType.IP, message = "登录尝试过于频繁，请60秒后再试")
    @PostMapping("/login")
    public Result<LoginResult> login(@RequestBody LoginForm form) {
        LoginResult loginResult = authService.login(form);
        return Result.success(loginResult);
    }

    @Operation(summary = "注销")
    @PostMapping("/logout")
    public Result<Void> logout() {
        authService.logout();
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
    public Result<Map<String, Object>> me() {
        return Result.success(authService.getAuthInfo());
    }

    @Operation(summary = "刷新令牌")
    @PostMapping("/refresh")
    public Result<LoginResult> refresh(@Valid @RequestBody RefreshTokenForm form) {
        return Result.success(authService.refreshToken(form));
    }
}
