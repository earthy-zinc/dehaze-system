package com.pei.dehaze.common.constant;

public interface SecurityConstants {

    String CAPTCHA_CODE_PREFIX = "captcha_code:";

    String ROLE_PERMS_PREFIX = "role:perms:";

    String SESSION_PREFIX = "session:";

    String SESSION_USER_PREFIX = "session:user:";

    String LOGIN_PATH = "/api/v1/auth/login";

    String ROLE_PREFIX = "ROLE_";

    String SESSION_COOKIE_NAME = "X-Session-Id";

    /** Session Redis TTL: 7天（秒） */
    long SESSION_TTL = 604800L;

    /** Session 滑动续期阈值: 剩余不足 1 天时自动续期（秒） */
    long RENEW_THRESHOLD = 86400L;

    /** 登录失败次数 Redis Key 前缀 */
    String LOGIN_FAIL_PREFIX = "login:fail:";

    /** 登录失败次数 Redis Key 前缀（IP 纬度） */
    String LOGIN_FAIL_IP_PREFIX = "login:fail:ip:";

    /** 最大登录失败次数 */
    int MAX_LOGIN_ATTEMPTS = 5;

    /** 登录锁定时间（分钟） */
    int LOCK_DURATION_MINUTES = 30;

}
