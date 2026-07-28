package com.pei.dehaze.service;

public interface LoginLogService {

    void recordLogin(Long userId, String username, String ip, int status, String message, String browser, String os, String location);
}
