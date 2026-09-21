package com.pei.dehaze.service.impl;

import cn.hutool.captcha.AbstractCaptcha;
import cn.hutool.captcha.CaptchaUtil;
import cn.hutool.captcha.generator.CodeGenerator;
import cn.hutool.core.text.CharSequenceUtil;
import cn.hutool.core.util.IdUtil;
import cn.hutool.http.useragent.UserAgent;
import cn.hutool.http.useragent.UserAgentUtil;
import cn.hutool.json.JSONArray;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.pei.dehaze.common.constant.SecurityConstants;
import com.pei.dehaze.common.enums.CaptchaTypeEnum;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.SysUserMapper;
import com.pei.dehaze.model.dto.CaptchaResult;
import com.pei.dehaze.model.form.LoginForm;
import com.pei.dehaze.model.dto.LoginResult;
import com.pei.dehaze.model.form.RegisterForm;
import com.pei.dehaze.model.entity.SysRole;
import com.pei.dehaze.model.entity.SysUser;
import com.pei.dehaze.model.entity.SysUserRole;
import com.pei.dehaze.model.query.LoginLogQuery;
import com.pei.dehaze.model.vo.LoginLogVO;
import com.pei.dehaze.model.vo.UserSessionVO;
import com.pei.dehaze.plugin.captcha.CaptchaProperties;
import com.pei.dehaze.security.model.SysUserDetails;
import com.pei.dehaze.security.util.SecurityUtils;
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
import org.springframework.data.redis.core.Cursor;
import org.springframework.data.redis.core.ScanOptions;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.data.redis.core.ZSetOperations;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.BadCredentialsException;
import org.springframework.security.authentication.DisabledException;
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
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.List;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
@Slf4j
public class AuthServiceImpl implements AuthService {

    private static final DateTimeFormatter SESSION_TIME_FORMAT = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    private final AuthenticationManager authenticationManager;
    private final StringRedisTemplate redisTemplate;
    private final CodeGenerator codeGenerator;
    private final Font captchaFont;
    private final CaptchaProperties captchaProperties;
    private final PasswordEncoder passwordEncoder;
    private final SysUserService sysUserService;
    private final SysUserMapper sysUserMapper;
    private final SysRoleService sysRoleService;
    private final SysUserRoleService sysUserRoleService;
    private final LoginLogService loginLogService;
    private final MemberService memberService;

    @Value("${system.use-multi-point:false}")
    private boolean useMultiPoint;

    @Override
    public LoginResult login(LoginForm form) {
        String username = form.getUsername().toLowerCase().trim();
        String deviceType = CharSequenceUtil.blankToDefault(form.getDeviceType(), "web");

        String cacheKey = SecurityConstants.CAPTCHA_CODE_PREFIX + form.getCaptchaKey();
        String cacheVerifyCode = redisTemplate.opsForValue().getAndDelete(cacheKey);
        if (cacheVerifyCode == null) {
            recordLogin(null, username, 0, "验证码已过期", deviceType);
            throw new BusinessException(ResultCode.VERIFY_CODE_TIMEOUT);
        }
        if (!codeGenerator.verify(cacheVerifyCode, form.getCaptchaCode())) {
            recordLogin(null, username, 0, "验证码错误", deviceType);
            throw new BusinessException(ResultCode.VERIFY_CODE_ERROR);
        }

        // ———— IP 纬度锁定检查 ————
        String clientIp = getCurrentClientIp();
        String ipFailKey = SecurityConstants.LOGIN_FAIL_IP_PREFIX + clientIp;
        String ipFailCountStr = redisTemplate.opsForValue().get(ipFailKey);
        int ipFailCount = parseIntSafe(ipFailCountStr);
        if (ipFailCount >= SecurityConstants.MAX_LOGIN_ATTEMPTS) {
            String msg = "IP登录失败次数过多，已临时锁定，请稍后重试";
            recordLogin(null, username, 0, msg, deviceType);
            throw new BusinessException(ResultCode.PASSWORD_ENTER_EXCEED_LIMIT, msg);
        }

        // ———— 用户名纬度锁定检查 ————
        String failKey = SecurityConstants.LOGIN_FAIL_PREFIX + username;
        String failCountStr = redisTemplate.opsForValue().get(failKey);
        Integer failCount = failCountStr != null ? Integer.parseInt(failCountStr) : null;
        if (failCount != null && failCount >= SecurityConstants.MAX_LOGIN_ATTEMPTS) {
            String msg = "账户已被锁定，请" + SecurityConstants.LOCK_DURATION_MINUTES + "分钟后再试";
            recordLogin(null, username, 0, msg, deviceType);
            throw new BusinessException(ResultCode.PASSWORD_ENTER_EXCEED_LIMIT, msg);
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
                recordLogin(null, username, 0, msg, deviceType);
                throw new BusinessException(ResultCode.USERNAME_OR_PASSWORD_ERROR, msg);
            } else {
                String msg = "账户已被锁定，请" + SecurityConstants.LOCK_DURATION_MINUTES + "分钟后再试";
                recordLogin(null, username, 0, msg, deviceType);
                throw new BusinessException(ResultCode.PASSWORD_ENTER_EXCEED_LIMIT, msg);
            }
        } catch (DisabledException e) {
            // 禁用用户与 Python/Go 对齐：A0202 账户被冻结，且不计入锁定失败计数
            String msg = "用户已被禁用";
            recordLogin(null, username, 0, msg, deviceType);
            throw new BusinessException(ResultCode.USER_ACCOUNT_LOCKED, msg);
        } catch (Exception e) {
            // loadUserByUsername 抛出的业务异常会被 Spring Security 包装（InternalAuthenticationServiceException），
            // 此处还原真实错误码，避免笼统报"认证服务暂时不可用"掩盖根因
            if (e.getCause() instanceof BusinessException biz) {
                recordLogin(null, username, 0, biz.getMessage(), deviceType);
                throw biz;
            }
            log.error("认证过程发生非凭证类异常，未递增失败计数: username={}", username, e);
            recordLogin(null, username, 0, "认证服务暂时不可用，请稍后重试", deviceType);
            throw new BusinessException(ResultCode.SYSTEM_EXECUTION_ERROR, "认证服务暂时不可用，请稍后重试");
        }

        redisTemplate.delete(failKey);
        redisTemplate.delete(ipFailKey);

        SysUserDetails userDetails = (SysUserDetails) authentication.getPrincipal();
        recordLogin(userDetails.getUserId(), username, 1, "登录成功", deviceType);

        // 会员档案兜底：种子账号与后台创建的用户不走注册流程，登录时确保
        // sys_member 行存在（否则计费配额校验 fail-closed 误报"配额不足"）
        memberService.ensureMemberProfile(userDetails.getUserId());

        String sessionId = IdUtil.fastSimpleUUID();

        // 多点登录控制：超限踢出最早登录的会话（不区分设备类型）
        if (useMultiPoint) {
            enforceDeviceLimit(sessionId, userDetails.getUserId(), userDetails.getAuthorities());
        }

        JSONObject session = new JSONObject();
        session.set("userId", userDetails.getUserId());
        session.set("username", userDetails.getUsername());
        session.set("deptId", userDetails.getDeptId());
        session.set("dataScope", userDetails.getDataScope());
        session.set("nickname", userDetails.getNickname());
        // 会话 authorities = ROLE_* 角色 + 权限标识，与 python 登录写会话同构：
        // python 从同一 Redis 会话派生 permissions，缺失权限标识会让代理到 python 的端点误判无权
        List<String> authorities = userDetails.getAuthorities().stream()
                .map(GrantedAuthority::getAuthority)
                .collect(Collectors.toList());
        if (userDetails.getPerms() != null) {
            authorities.addAll(userDetails.getPerms());
        }
        session.set("authorities", authorities);
        // 会话管理（F-AM-011）所需元数据：设备类型/登录IP/登录与最后访问时间
        String now = LocalDateTime.now().format(SESSION_TIME_FORMAT);
        session.set("deviceType", deviceType);
        session.set("loginIp", clientIp);
        session.set("loginTime", now);
        session.set("lastAccessTime", now);

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
        String cacheVerifyCode = redisTemplate.opsForValue().getAndDelete(cacheKey);
        if (cacheVerifyCode == null) {
            throw new BusinessException(ResultCode.VERIFY_CODE_TIMEOUT);
        }
        if (!codeGenerator.verify(cacheVerifyCode, form.getCaptchaCode())) {
            throw new BusinessException(ResultCode.VERIFY_CODE_ERROR);
        }

        String username = form.getUsername().toLowerCase().trim();

        // 业务白名单：用户名查全表判重（含软删行），删除后永久不可复用，理由见 SysUserMapper#countByUsernameAllDeleted
        long userCount = sysUserMapper.countByUsernameAllDeleted(username);
        if (userCount > 0) {
            throw new BusinessException(ResultCode.DATA_EXISTS, "用户名已被注册");
        }

        SysUser user = new SysUser();
        user.setUsername(username);
        user.setNickname(form.getNickname().trim());
        user.setPassword(passwordEncoder.encode(form.getPassword()));
        user.setGender(1);
        user.setStatus(1);
        user.setDeleted(0L);
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

        // 注册签发的会话同样登记进设备数索引，否则该会话不占额度（与登录路径一致）
        if (useMultiPoint) {
            enforceDeviceLimit(sessionId, userDetails.getUserId(), userDetails.getAuthorities());
        }

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
            String sessionJson = redisTemplate.opsForValue().get(SecurityConstants.SESSION_PREFIX + sessionId);
            if (sessionJson != null) {
                JSONObject session = JSONUtil.parseObj(sessionJson);
                Long userId = session.getLong("userId");
                // 索引（session:user:{userId} ZSet）剔除本会话元素，其他端在线会话不受影响
                if (userId != null) {
                    redisTemplate.opsForZSet().remove(SecurityConstants.SESSION_USER_PREFIX + userId, sessionId);
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
     * 多点登录控制（F-AM-011）：按同时在线设备数上限踢出最早登录的会话。
     * <p>
     * 索引 session:user:{userId} 为 ZSet（member=sessionId，score=登录 epoch 秒），三端共享同一
     * Redis 结构。超限时新会话保留，最早的若干会话被删除（session:{sessionId} + 索引元素），
     * 其下一次请求因会话不存在而返回 401。管理员（ROOT/ADMIN）固定 10 台。
     */
    void enforceDeviceLimit(String newSessionId, Long userId, Collection<? extends GrantedAuthority> authorities) {
        boolean adminSession = authorities.stream()
                .map(GrantedAuthority::getAuthority)
                .anyMatch(authority -> (SecurityConstants.ROLE_PREFIX + "ROOT").equals(authority)
                        || (SecurityConstants.ROLE_PREFIX + "ADMIN").equals(authority));
        int maxDevices = adminSession ? SecurityConstants.ADMIN_MAX_DEVICES : memberService.getMaxDevices(userId);

        String indexKey = SecurityConstants.SESSION_USER_PREFIX + userId;
        ZSetOperations<String, String> zset = redisTemplate.opsForZSet();
        zset.add(indexKey, newSessionId, (double) (System.currentTimeMillis() / 1000));
        redisTemplate.expire(indexKey, SecurityConstants.SESSION_TTL, TimeUnit.SECONDS);

        Long total = zset.zCard(indexKey);
        if (total == null || total <= maxDevices) {
            return;
        }
        Set<String> members = zset.range(indexKey, 0, -1);
        if (members == null) {
            return;
        }
        int excess = (int) (total - maxDevices);
        List<String> evicted = new ArrayList<>(excess);
        for (String sessionId : members) {
            // 排除本次新会话：同秒登录时 score 相同，按 member 字典序排序也可能把它排在前面
            if (sessionId.equals(newSessionId)) {
                continue;
            }
            evicted.add(sessionId);
            if (evicted.size() == excess) {
                break;
            }
        }
        if (evicted.isEmpty()) {
            return;
        }
        redisTemplate.delete(evicted.stream().map(id -> SecurityConstants.SESSION_PREFIX + id).toList());
        zset.remove(indexKey, evicted.toArray());
    }

    private void recordLogin(Long userId, String username, int status, String message, String deviceType) {
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
            loginLogService.recordLogin(userId, username, ip, status, message, browser, os, deviceType);
        } catch (Exception e) {
            log.warn("记录登录日志失败: username={}, status={}", username, status, e);
        }
    }

    @Override
    public IPage<LoginLogVO> listLoginLogs(LoginLogQuery query) {
        // 管理员（ROOT/ADMIN）可查看全量日志，普通用户仅查看本人日志（对齐 python auth_service.list_login_logs）
        Long restrictedUserId = SecurityUtils.isAdmin() ? null : SecurityUtils.getUserId();
        return loginLogService.pageLoginLogs(query, restrictedUserId);
    }

    @Override
    public List<UserSessionVO> listSessions(String username) {
        List<UserSessionVO> sessions = new ArrayList<>();
        try (Cursor<String> cursor = redisTemplate.scan(ScanOptions.scanOptions()
                .match(SecurityConstants.SESSION_PREFIX + "*").build())) {
            while (cursor.hasNext()) {
                String key = cursor.next();
                if (key.startsWith(SecurityConstants.SESSION_USER_PREFIX)) {
                    continue;
                }
                String raw = redisTemplate.opsForValue().get(key);
                if (raw == null) {
                    continue;
                }
                JSONObject data = JSONUtil.parseObj(raw);
                if (!username.equals(data.getStr("username"))) {
                    continue;
                }
                String loginTime = CharSequenceUtil.nullToEmpty(data.getStr("loginTime"));
                sessions.add(UserSessionVO.builder()
                        .sessionId(key.substring(SecurityConstants.SESSION_PREFIX.length()))
                        .username(CharSequenceUtil.nullToEmpty(data.getStr("username")))
                        .deviceType(CharSequenceUtil.blankToDefault(data.getStr("deviceType"), "web"))
                        .loginTime(loginTime)
                        .ip(CharSequenceUtil.nullToEmpty(data.getStr("loginIp")))
                        .lastAccessTime(CharSequenceUtil.blankToDefault(data.getStr("lastAccessTime"), loginTime))
                        .build());
            }
        }
        sessions.sort(Comparator.comparing(UserSessionVO::getLoginTime).reversed());
        return sessions;
    }

    @Override
    public void kickSession(String sessionId) {
        String key = SecurityConstants.SESSION_PREFIX + sessionId;
        String raw = redisTemplate.opsForValue().get(key);
        if (raw == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "会话不存在或已过期");
        }
        JSONObject data = JSONUtil.parseObj(raw);
        JSONArray authorities = data.getJSONArray("authorities");
        if (authorities != null && authorities.contains(SecurityConstants.ROLE_PREFIX + "ROOT")) {
            throw new BusinessException(ResultCode.OPERATION_NOT_ALLOW, "超级管理员会话不可被踢出");
        }
        redisTemplate.delete(key);
        // 同步清理多点登录索引（session:user:{userId} ZSet 中的本会话元素）
        Long userId = data.getLong("userId");
        if (userId != null) {
            redisTemplate.opsForZSet().remove(SecurityConstants.SESSION_USER_PREFIX + userId, sessionId);
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
