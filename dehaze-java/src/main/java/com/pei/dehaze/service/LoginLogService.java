package com.pei.dehaze.service;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.pei.dehaze.model.query.LoginLogQuery;
import com.pei.dehaze.model.vo.LoginLogVO;

public interface LoginLogService {

    void recordLogin(Long userId, String username, String ip, int status, String message, String browser, String os, String deviceType);

    /**
     * 分页查询登录日志（MongoDB）
     *
     * @param query            查询条件（用户名/IP/状态/设备类型/时间范围精确匹配）
     * @param restrictedUserId 非空时仅查询该用户的日志（普通用户数据权限限定）
     */
    IPage<LoginLogVO> pageLoginLogs(LoginLogQuery query, Long restrictedUserId);
}
