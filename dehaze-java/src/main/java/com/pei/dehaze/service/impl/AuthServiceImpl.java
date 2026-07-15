package com.pei.dehaze.service.impl;

import cn.hutool.captcha.AbstractCaptcha;
import cn.hutool.captcha.CaptchaUtil;
import cn.hutool.captcha.generator.CodeGenerator;
import cn.hutool.core.text.CharSequenceUtil;
import cn.hutool.core.util.IdUtil;
import cn.hutool.json.JSONObject;
import cn.hutool.jwt.JWTUtil;
import cn.hutool.jwt.RegisteredPayload;
import com.pei.dehaze.common.constant.SecurityConstants;
import com.pei.dehaze.common.enums.CaptchaTypeEnum;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.model.dto.CaptchaResult;
import com.pei.dehaze.model.dto.LoginForm;
import com.pei.dehaze.model.dto.LoginResult;
import com.pei.dehaze.plugin.captcha.CaptchaProperties;
import com.pei.dehaze.security.model.SysUserDetails;
import com.pei.dehaze.security.util.JwtUtils;
import com.pei.dehaze.service.AuthService;
import jakarta.servlet.http.HttpServletRequest;
import lombok.RequiredArgsConstructor;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.http.HttpHeaders;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Service;
import org.springframework.web.context.request.RequestContextHolder;
import org.springframework.web.context.request.ServletRequestAttributes;

import java.awt.*;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

/**
 * 认证服务实现类
 */
@Service
@RequiredArgsConstructor
public class AuthServiceImpl implements AuthService {

    private final AuthenticationManager authenticationManager;
    private final RedisTemplate<String, Object> redisTemplate;
    private final CodeGenerator codeGenerator;
    private final Font captchaFont;
    private final CaptchaProperties captchaProperties;
    private final JwtUtils jwtUtils;

    @Override
    public LoginResult login(LoginForm form) {
        // 1. 验证码校验
        String cacheKey = SecurityConstants.CAPTCHA_CODE_PREFIX + form.getCaptchaKey();
        String cacheVerifyCode = (String) redisTemplate.opsForValue().get(cacheKey);
        if (cacheVerifyCode == null) {
            throw new BusinessException(ResultCode.VERIFY_CODE_TIMEOUT);
        }
        if (!codeGenerator.verify(cacheVerifyCode, form.getCaptchaCode())) {
            throw new BusinessException(ResultCode.VERIFY_CODE_ERROR);
        }
        // 验证后删除验证码
        redisTemplate.delete(cacheKey);

        // 2. 用户认证
        UsernamePasswordAuthenticationToken authenticationToken =
                new UsernamePasswordAuthenticationToken(form.getUsername().toLowerCase().trim(), form.getPassword());
        Authentication authentication = authenticationManager.authenticate(authenticationToken);

        // 3. 生成 Token
        String accessToken = jwtUtils.createToken(authentication);

        // 4. 获取用户信息
        SysUserDetails userDetails = (SysUserDetails) authentication.getPrincipal();

        return buildLoginResult(accessToken, userDetails);
    }

    @Override
    public void logout() {
        ServletRequestAttributes requestAttributes = (ServletRequestAttributes) RequestContextHolder.getRequestAttributes();
        if (requestAttributes == null) throw new BusinessException("请求上下文为空");
        HttpServletRequest request = requestAttributes.getRequest();
        String token = request.getHeader(HttpHeaders.AUTHORIZATION);
        if (CharSequenceUtil.isNotBlank(token) && token.startsWith(SecurityConstants.JWT_TOKEN_PREFIX)) {
            token = token.substring(SecurityConstants.JWT_TOKEN_PREFIX.length());
            JSONObject payloads = JWTUtil.parseToken(token).getPayloads();
            String jti = payloads.getStr(RegisteredPayload.JWT_ID);
            Long expiration = payloads.getLong(RegisteredPayload.EXPIRES_AT);
            if (expiration != null) {
                long currentTimeSeconds = System.currentTimeMillis() / 1000;
                if (expiration < currentTimeSeconds) {
                    return;
                }
                long ttl = expiration - currentTimeSeconds;
                redisTemplate.opsForValue()
                        .set(SecurityConstants.BLACKLIST_TOKEN_PREFIX + jti, "", ttl, TimeUnit.SECONDS);
            } else {
                redisTemplate.opsForValue()
                        .set(SecurityConstants.BLACKLIST_TOKEN_PREFIX + jti, "");
            }
        }
        SecurityContextHolder.clearContext();
    }

    @Override
    public CaptchaResult getCaptcha() {
        String captchaType = captchaProperties.getType();
        AbstractCaptcha captcha = getAbstractCaptcha(captchaType);
        captcha.setGenerator(codeGenerator);
        captcha.setTextAlpha(captchaProperties.getTextAlpha());
        captcha.setFont(captchaFont);

        String captchaCode = captcha.getCode();
        String imageBase64Data = captcha.getImageBase64Data();

        String captchaKey = IdUtil.fastSimpleUUID();
        redisTemplate.opsForValue().set(SecurityConstants.CAPTCHA_CODE_PREFIX + captchaKey, captchaCode,
                captchaProperties.getExpireSeconds(), TimeUnit.SECONDS);

        return CaptchaResult.builder()
                .captchaKey(captchaKey)
                .captchaBase64(imageBase64Data)
                .build();
    }

    @Override
    public Map<String, Object> getAuthInfo() {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        if (authentication == null || !(authentication.getPrincipal() instanceof SysUserDetails)) {
            throw new BusinessException(ResultCode.TOKEN_INVALID);
        }
        SysUserDetails userDetails = (SysUserDetails) authentication.getPrincipal();

        Map<String, Object> result = new HashMap<>();
        result.put("userId", userDetails.getUserId());
        result.put("username", userDetails.getUsername());
        result.put("nickname", userDetails.getNickname());

        // 角色列表（去掉 ROLE_ 前缀）
        result.put("roles", authentication.getAuthorities().stream()
                .map(GrantedAuthority::getAuthority)
                .map(a -> a.startsWith(SecurityConstants.ROLE_PREFIX) ? a.substring(SecurityConstants.ROLE_PREFIX.length()) : a)
                .collect(Collectors.toList()));

        // 权限列表
        result.put("permissions", userDetails.getPerms() != null
                ? new ArrayList<>(userDetails.getPerms())
                : Collections.emptyList());

        return result;
    }

    @Override
    public LoginResult refreshToken() {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        if (authentication == null || !(authentication.getPrincipal() instanceof SysUserDetails)) {
            throw new BusinessException(ResultCode.TOKEN_INVALID);
        }

        String accessToken = jwtUtils.createToken(authentication);
        SysUserDetails userDetails = (SysUserDetails) authentication.getPrincipal();

        return buildLoginResult(accessToken, userDetails);
    }

    private AbstractCaptcha getAbstractCaptcha(String captchaType) {
        int width = captchaProperties.getWidth();
        int height = captchaProperties.getHeight();
        int interfereCount = captchaProperties.getInterfereCount();
        int codeLength = captchaProperties.getCode().getLength();

        return switch (CaptchaTypeEnum.valueOf(captchaType.toUpperCase())) {
            case CIRCLE -> CaptchaUtil.createCircleCaptcha(width, height, codeLength, interfereCount);
            case GIF -> CaptchaUtil.createGifCaptcha(width, height, codeLength);
            case LINE -> CaptchaUtil.createLineCaptcha(width, height, codeLength, interfereCount);
            case SHEAR -> CaptchaUtil.createShearCaptcha(width, height, codeLength, interfereCount);
        };
    }

    /**
     * 构建登录结果
     */
    private LoginResult buildLoginResult(String accessToken, SysUserDetails userDetails) {
        return LoginResult.builder()
                .tokenType(SecurityConstants.TOKEN_TYPE)
                .accessToken(accessToken)
                .user(LoginResult.UserInfo.builder()
                        .id(userDetails.getUserId())
                        .username(userDetails.getUsername())
                        .nickname(userDetails.getNickname())
                        .build())
                .build();
    }
}
