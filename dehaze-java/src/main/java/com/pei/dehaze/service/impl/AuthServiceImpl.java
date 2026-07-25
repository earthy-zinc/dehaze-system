package com.pei.dehaze.service.impl;

import cn.hutool.captcha.AbstractCaptcha;
import cn.hutool.captcha.CaptchaUtil;
import cn.hutool.captcha.generator.CodeGenerator;
import cn.hutool.core.util.IdUtil;
import cn.hutool.json.JSONObject;
import com.pei.dehaze.common.constant.SecurityConstants;
import com.pei.dehaze.common.enums.CaptchaTypeEnum;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.model.dto.CaptchaResult;
import com.pei.dehaze.model.dto.LoginForm;
import com.pei.dehaze.model.dto.LoginResult;
import com.pei.dehaze.plugin.captcha.CaptchaProperties;
import com.pei.dehaze.security.model.SysUserDetails;
import com.pei.dehaze.service.AuthService;
import jakarta.servlet.http.Cookie;
import jakarta.servlet.http.HttpServletRequest;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.BadCredentialsException;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;
import org.springframework.web.context.request.RequestContextHolder;
import org.springframework.web.context.request.ServletRequestAttributes;

import java.awt.*;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
@Slf4j
public class AuthServiceImpl implements AuthService {

    private final AuthenticationManager authenticationManager;
    private final StringRedisTemplate redisTemplate;
    private final CodeGenerator codeGenerator;
    private final Font captchaFont;
    private final CaptchaProperties captchaProperties;

    private static final String LOGIN_FAIL_PREFIX = "login:fail:";
    private static final int MAX_LOGIN_ATTEMPTS = 5;
    private static final int LOCK_DURATION_MINUTES = 30;
    private static final long SESSION_TTL = 604800L;

    @Override
    public LoginResult login(LoginForm form) {
        String cacheKey = SecurityConstants.CAPTCHA_CODE_PREFIX + form.getCaptchaKey();
        String cacheVerifyCode = redisTemplate.opsForValue().get(cacheKey);
        if (cacheVerifyCode == null) {
            throw new BusinessException(ResultCode.VERIFY_CODE_TIMEOUT);
        }
        if (!codeGenerator.verify(cacheVerifyCode, form.getCaptchaCode())) {
            throw new BusinessException(ResultCode.VERIFY_CODE_ERROR);
        }
        redisTemplate.delete(cacheKey);

        String username = form.getUsername().toLowerCase().trim();
        String failKey = LOGIN_FAIL_PREFIX + username;
        String failCountStr = redisTemplate.opsForValue().get(failKey);
        Integer failCount = failCountStr != null ? Integer.parseInt(failCountStr) : null;
        if (failCount != null && failCount >= MAX_LOGIN_ATTEMPTS) {
            throw new BusinessException("账户已被锁定，请" + LOCK_DURATION_MINUTES + "分钟后再试");
        }

        UsernamePasswordAuthenticationToken authenticationToken =
                new UsernamePasswordAuthenticationToken(username, form.getPassword());
        Authentication authentication;
        try {
            authentication = authenticationManager.authenticate(authenticationToken);
        } catch (BadCredentialsException | UsernameNotFoundException e) {
            Long count = redisTemplate.opsForValue().increment(failKey);
            if (count != null && count == 1) {
                redisTemplate.expire(failKey, LOCK_DURATION_MINUTES, TimeUnit.MINUTES);
            }
            long remaining = MAX_LOGIN_ATTEMPTS - (count != null ? count : 0);
            if (remaining > 0) {
                throw new BusinessException("用户名或密码错误，剩余" + remaining + "次尝试机会");
            } else {
                throw new BusinessException("账户已被锁定，请" + LOCK_DURATION_MINUTES + "分钟后再试");
            }
        } catch (Exception e) {
            log.error("认证过程发生非凭证类异常，未递增失败计数: username={}", username, e);
            throw new BusinessException("认证服务暂时不可用，请稍后重试");
        }

        redisTemplate.delete(failKey);

        SysUserDetails userDetails = (SysUserDetails) authentication.getPrincipal();

        String sessionId = IdUtil.fastSimpleUUID();
        JSONObject session = new JSONObject();
        session.set("userId", userDetails.getUserId());
        session.set("username", userDetails.getUsername());
        session.set("deptId", userDetails.getDeptId());
        session.set("dataScope", userDetails.getDataScope());
        session.set("nickname", userDetails.getNickname());
        List<String> authorities = userDetails.getAuthorities().stream()
                .map(GrantedAuthority::getAuthority)
                .collect(Collectors.toList());
        session.set("authorities", authorities);

        redisTemplate.opsForValue().set(
                SecurityConstants.SESSION_PREFIX + sessionId,
                session.toString(),
                SESSION_TTL,
                TimeUnit.SECONDS);

        return LoginResult.builder()
                .sessionId(sessionId)
                .user(LoginResult.UserInfo.builder()
                        .id(userDetails.getUserId())
                        .username(userDetails.getUsername())
                        .nickname(userDetails.getNickname())
                        .build())
                .build();
    }

    @Override
    public void logout() {
        ServletRequestAttributes requestAttributes = (ServletRequestAttributes) RequestContextHolder.getRequestAttributes();
        if (requestAttributes == null) throw new BusinessException("请求上下文为空");
        HttpServletRequest request = requestAttributes.getRequest();

        String sessionId = null;
        Cookie[] cookies = request.getCookies();
        if (cookies != null) {
            for (Cookie cookie : cookies) {
                if (SecurityConstants.SESSION_COOKIE_NAME.equals(cookie.getName())) {
                    sessionId = cookie.getValue();
                    break;
                }
            }
        }
        if (sessionId == null) {
            sessionId = request.getHeader(SecurityConstants.SESSION_COOKIE_NAME);
        }
        if (sessionId != null) {
            redisTemplate.delete(SecurityConstants.SESSION_PREFIX + sessionId);
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
}
