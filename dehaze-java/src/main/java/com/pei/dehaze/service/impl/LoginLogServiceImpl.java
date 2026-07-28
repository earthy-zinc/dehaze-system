package com.pei.dehaze.service.impl;

import com.pei.dehaze.model.entity.LoginLog;
import com.pei.dehaze.repository.LoginLogRepository;
import com.pei.dehaze.service.LoginLogService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;

@Slf4j
@Service
@RequiredArgsConstructor
public class LoginLogServiceImpl implements LoginLogService {

    private final LoginLogRepository loginLogRepository;

    @Async("datasetTaskExecutor")
    @Override
    public void recordLogin(Long userId, String username, String ip, int status, String message, String browser, String os, String location) {
        try {
            LoginLog loginLog = new LoginLog();
            loginLog.setUserId(userId);
            loginLog.setUsername(username);
            loginLog.setIp(ip);
            loginLog.setLocation(location);
            loginLog.setBrowser(browser);
            loginLog.setOs(os);
            loginLog.setStatus(status);
            loginLog.setMessage(message);
            loginLog.setCreateTime(LocalDateTime.now());
            loginLogRepository.save(loginLog);
        } catch (Exception e) {
            log.warn("写入登录日志失败: username={}, status={}", username, status, e);
        }
    }
}
