package com.pei.clarity.service;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.pei.clarity.model.dto.CaptchaResult;
import com.pei.clarity.model.dto.LoginResult;
import com.pei.clarity.model.form.LoginForm;
import com.pei.clarity.model.form.RegisterForm;
import com.pei.clarity.model.query.LoginLogQuery;
import com.pei.clarity.model.vo.LoginLogVO;
import com.pei.clarity.model.vo.UserSessionVO;
import java.util.List;

public interface AuthService {

    LoginResult login(LoginForm form);

    LoginResult register(RegisterForm form);

    void logout();

    CaptchaResult getCaptcha();

    /**
     * 分页查询登录日志（管理员全量，普通用户仅本人）
     */
    IPage<LoginLogVO> listLoginLogs(LoginLogQuery query);

    /**
     * 查询指定用户的在线会话列表
     */
    List<UserSessionVO> listSessions(String username);

    /**
     * 踢出指定在线会话（超级管理员会话受保护）
     */
    void kickSession(String sessionId);
}
