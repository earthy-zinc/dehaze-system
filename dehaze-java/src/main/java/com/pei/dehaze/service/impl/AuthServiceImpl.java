package com.pei.dehaze.service.impl;

import cn.hutool.captcha.AbstractCaptcha;
import cn.hutool.captcha.CaptchaUtil;
import cn.hutool.captcha.generator.CodeGenerator;
import cn.hutool.core.text.CharSequenceUtil;
import cn.hutool.core.util.IdUtil;
import cn.hutool.http.useragent.UserAgent;
import cn.hutool.http.useragent.UserAgentUtil;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.pei.dehaze.common.constant.SecurityConstants;
import com.pei.dehaze.common.enums.CaptchaTypeEnum;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.model.dto.CaptchaResult;
import com.pei.dehaze.model.form.LoginForm;
import com.pei.dehaze.model.dto.LoginResult;
import com.pei.dehaze.model.form.RegisterForm;
import com.pei.dehaze.model.entity.SysRole;
import com.pei.dehaze.model.entity.SysUser;
import com.pei.dehaze.model.entity.SysUserRole;
import com.pei.dehaze.plugin.captcha.CaptchaProperties;
import com.pei.dehaze.security.model.SysUserDetails;
import com.pei.dehaze.service.AuthService;
import com.pei.dehaze.service.LoginLogService;
import com.pei.dehaze.service.MemberService;
import com.pei.dehaze.service.SysRoleService;
import com.pei.dehaze.service.SysUserRoleService;
import com.pei.dehaze.service.SysUserService;
import jakarta.servlet.http.Cookie;
import jakarta.servlet.http.HttpServletRequest;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.BadCredentialsException;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.web.context.request.RequestContextHolder;
import org.springframework.web.context.request.ServletRequestAttributes;

import java.awt.*;
import java.util.List;
import java.util.Set;
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
    private final PasswordEncoder passwordEncoder;
    private final SysUserService sysUserService;
    private final SysRoleService sysRoleService;
    private final SysUserRoleService sysUserRoleService;
    private final LoginLogService loginLogService;
    private final MemberService memberService;

    @Value("${system.use-multi-point:false}")
    private boolean useMultiPoint;

    @Override
    public LoginResult login(LoginForm form) {
        String username = form.getUsername().toLowerCase().trim();

        String cacheKey = SecurityConstants.CAPTCHA_CODE_PREFIX + form.getCaptchaKey();
        String cacheVerifyCode = redisTemplate.opsForValue().get(cacheKey);
        if (cacheVerifyCode == null) {
            recordLogin(null, username, 0, "验证码已过期");
            throw new BusinessException(ResultCode.VERIFY_CODE_TIMEOUT);
        }
        if (!codeGenerator.verify(cacheVerifyCode, form.getCaptchaCode())) {
            recordLogin(null, username, 0, "验证码错误");
            throw new BusinessException(ResultCode.VERIFY_CODE_ERROR);
        }
        redisTemplate.delete(cacheKey);

        // ———— IP 纬度锁定检查 ————
        String clientIp = getCurrentClientIp();
        String ipFailKey = SecurityConstants.LOGIN_FAIL_IP_PREFIX + clientIp;
        String ipFailCountStr = redisTemplate.opsForValue().get(ipFailKey);
        int ipFailCount = parseIntSafe(ipFailCountStr);
        if (ipFailCount >= SecurityConstants.MAX_LOGIN_ATTEMPTS) {
            String msg = "IP登录失败次数过多，已临时锁定，请稍后重试";
            recordLogin(null, username, 0, msg);
            throw new BusinessException(msg);
        }

        // ———— 用户名纬度锁定检查 ————
        String failKey = SecurityConstants.LOGIN_FAIL_PREFIX + username;
        String failCountStr = redisTemplate.opsForValue().get(failKey);
        Integer failCount = failCountStr != null ? Integer.parseInt(failCountStr) : null;
        if (failCount != null && failCount >= SecurityConstants.MAX_LOGIN_ATTEMPTS) {
            String msg = "账户已被锁定，请" + SecurityConstants.LOCK_DURATION_MINUTES + "分钟后再试";
            recordLogin(null, username, 0, msg);
            throw new BusinessException(msg);
        }

        UsernamePasswordAuthenticationToken authenticationToken =
                new UsernamePasswordAuthenticationToken(username, form.getPassword());
        Authentication authentication;
        try {
            authentication = authenticationManager.authenticate(authenticationToken);
        } catch (BadCredentialsException | UsernameNotFoundException e) {
            // 同时递增 IP 和用户名纬度失败计数
            incrementFailCount(ipFailKey);
            Long count = redisTemplate.opsForValue().increment(failKey);
            if (count != null && count == 1) {
                redisTemplate.expire(failKey, SecurityConstants.LOCK_DURATION_MINUTES, TimeUnit.MINUTES);
            }
            long remaining = SecurityConstants.MAX_LOGIN_ATTEMPTS - (count != null ? count : 0);
            if (remaining > 0) {
                String msg = "用户名或密码错误，剩余" + remaining + "次尝试机会";
                recordLogin(null, username, 0, msg);
                throw new BusinessException(msg);
            } else {
                String msg = "账户已被锁定，请" + SecurityConstants.LOCK_DURATION_MINUTES + "分钟后再试";
                recordLogin(null, username, 0, msg);
                throw new BusinessException(msg);
            }
        } catch (Exception e) {
            log.error("认证过程发生非凭证类异常，未递增失败计数: username={}", username, e);
            recordLogin(null, username, 0, "认证服务暂时不可用，请稍后重试");
            throw new BusinessException("认证服务暂时不可用，请稍后重试");
        }

        redisTemplate.delete(failKey);
        redisTemplate.delete(ipFailKey);

        SysUserDetails userDetails = (SysUserDetails) authentication.getPrincipal();
        recordLogin(userDetails.getUserId(), username, 1, "登录成功");

        String sessionId = IdUtil.fastSimpleUUID();

        // 多点登录控制：删除旧 Session，仅保留最新
        if (useMultiPoint) {
            handleMultiPointSession(sessionId, username);
        }

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
                SecurityConstants.SESSION_TTL,
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
    public LoginResult register(RegisterForm form) {
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

        long userCount = sysUserService.count(new LambdaQueryWrapper<SysUser>()
                .eq(SysUser::getUsername, username)
                .eq(SysUser::getDeleted, 0));
        if (userCount > 0) {
            throw new BusinessException(ResultCode.DATA_EXISTS, "用户名已被注册");
        }

        SysUser user = new SysUser();
        user.setUsername(username);
        user.setNickname(form.getNickname().trim());
        user.setPassword(passwordEncoder.encode(form.getPassword()));
        user.setGender(1);
        user.setStatus(1);
        user.setDeleted(0);
        sysUserService.save(user);

        SysRole guestRole = sysRoleService.getOne(new LambdaQueryWrapper<SysRole>()
                .eq(SysRole::getCode, "GUEST")
                .eq(SysRole::getStatus, 1)
                .eq(SysRole::getDeleted, 0));
        if (guestRole != null) {
            SysUserRole userRole = new SysUserRole(user.getId(), guestRole.getId());
            sysUserRoleService.save(userRole);
        }

        memberService.initMember(user.getId());

        SysUserDetails userDetails = new SysUserDetails();
        userDetails.setUserId(user.getId());
        userDetails.setUsername(user.getUsername());
        userDetails.setNickname(user.getNickname());
        userDetails.setDeptId(null);
        userDetails.setDataScope(guestRole != null ? guestRole.getDataScope() : null);
        userDetails.setAuthorities(guestRole != null
                ? Set.of(new SimpleGrantedAuthority("ROLE_GUEST"))
                : Set.of());

        String sessionId = IdUtil.fastSimpleUUID();
        JSONObject session = new JSONObject();
        session.set("userId", userDetails.getUserId());
        session.set("username", userDetails.getUsername());
        session.set("nickname", userDetails.getNickname());
        session.set("deptId", userDetails.getDeptId());
        session.set("dataScope", userDetails.getDataScope());
        List<String> authorities = userDetails.getAuthorities().stream()
                .map(GrantedAuthority::getAuthority)
                .collect(Collectors.toList());
        session.set("authorities", authorities);

        redisTemplate.opsForValue().set(
                SecurityConstants.SESSION_PREFIX + sessionId,
                session.toString(),
                SecurityConstants.SESSION_TTL,
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
            // 获取 username 以清理多点登录索引
            String sessionJson = redisTemplate.opsForValue().get(SecurityConstants.SESSION_PREFIX + sessionId);
            if (sessionJson != null) {
                JSONObject session = JSONUtil.parseObj(sessionJson);
                String username = session.getStr("username");
                if (username != null) {
                    redisTemplate.delete(SecurityConstants.SESSION_USER_PREFIX + username);
                }
            }
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

    private void incrementFailCount(String key) {
        Long count = redisTemplate.opsForValue().increment(key);
        if (count != null && count == 1) {
            redisTemplate.expire(key, SecurityConstants.LOCK_DURATION_MINUTES, TimeUnit.MINUTES);
        }
    }

    private int parseIntSafe(String str) {
        if (str == null || str.isEmpty()) return 0;
        try {
            return Integer.parseInt(str);
        } catch (NumberFormatException e) {
            return 0;
        }
    }

    private String getCurrentClientIp() {
        ServletRequestAttributes requestAttributes =
                (ServletRequestAttributes) RequestContextHolder.getRequestAttributes();
        if (requestAttributes == null) return "unknown";
        return getClientIp(requestAttributes.getRequest());
    }

    /**
     * 多点登录控制：删除同一用户名下的旧 Session，仅保留最新。
     * 通过 Redis key session:user:{username} 记录当前活跃 Session ID。
     */
    private void handleMultiPointSession(String newSessionId, String username) {
        String userSessionKey = SecurityConstants.SESSION_USER_PREFIX + username;
        String oldSessionId = redisTemplate.opsForValue().get(userSessionKey);
        if (oldSessionId != null && !oldSessionId.isEmpty()) {
            redisTemplate.delete(SecurityConstants.SESSION_PREFIX + oldSessionId);
            log.info("多点登录：已删除旧Session, username={}, oldSessionId={}", username, oldSessionId);
        }
        redisTemplate.opsForValue().set(userSessionKey, newSessionId,
                SecurityConstants.SESSION_TTL, TimeUnit.SECONDS);
    }

    private void recordLogin(Long userId, String username, int status, String message) {
        try {
            ServletRequestAttributes requestAttributes = (ServletRequestAttributes) RequestContextHolder.getRequestAttributes();
            String ip = null;
            String browser = null;
            String os = null;
            if (requestAttributes != null) {
                HttpServletRequest request = requestAttributes.getRequest();
                ip = getClientIp(request);
                String userAgent = request.getHeader("User-Agent");
                if (CharSequenceUtil.isNotBlank(userAgent)) {
                    UserAgent ua = UserAgentUtil.parse(userAgent);
                    browser = ua.getBrowser() != null ? ua.getBrowser().getName() : null;
                    os = ua.getOs() != null ? ua.getOs().getName() : null;
                }
            }
            loginLogService.recordLogin(userId, username, ip, status, message, browser, os, null);
        } catch (Exception e) {
            log.warn("记录登录日志失败: username={}, status={}", username, status, e);
        }
    }

    private String getClientIp(HttpServletRequest request) {
        String ip = request.getHeader("X-Forwarded-For");
        if (CharSequenceUtil.isBlank(ip) || "unknown".equalsIgnoreCase(ip)) {
            ip = request.getHeader("X-Real-IP");
        }
        if (CharSequenceUtil.isBlank(ip) || "unknown".equalsIgnoreCase(ip)) {
            ip = request.getRemoteAddr();
        }
        if (ip != null && ip.contains(",")) {
            ip = ip.split(",")[0].trim();
        }
        return ip;
    }
}
