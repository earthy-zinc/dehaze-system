package com.pei.dehaze.utils;

import java.util.regex.Pattern;

/**
 * 输入验证工具类，用于验证用户名、密码、邮箱等输入内容的有效性
 */
public class InputValidator {
    // 用户名正则表达式：4-20位字母、数字、下划线
    private static final Pattern USERNAME_PATTERN = Pattern.compile("^[a-zA-Z0-9_]{4,20}$");
    // 密码正则表达式：8-20位，至少包含字母和数字
    private static final Pattern PASSWORD_PATTERN = Pattern.compile("^(?=.*[A-Za-z])(?=.*\\d)[A-Za-z\\d]{8,20}$");
    // 邮箱正则表达式
    private static final Pattern EMAIL_PATTERN = Pattern.compile("^[a-zA-Z0-9_+&*-]+(?:\\.[a-zA-Z0-9_+&*-]+)*@(?:[a-zA-Z0-9-]+\\.)+[a-zA-Z]{2,7}$");
    // 昵称正则表达式：2-20位任意字符，不能包含特殊符号
    private static final Pattern NICKNAME_PATTERN = Pattern.compile("^[\u4e00-\u9fa5a-zA-Z0-9_]{2,20}$");

    /**
     * 验证用户名是否有效
     */
    public static boolean isValidUsername(String username) {
        if (username == null || username.isEmpty()) {
            return false;
        }
        return USERNAME_PATTERN.matcher(username).matches();
    }

    /**
     * 验证密码是否有效
     */
    public static boolean isValidPassword(String password) {
        if (password == null || password.isEmpty()) {
            return false;
        }
        return PASSWORD_PATTERN.matcher(password).matches();
    }

    /**
     * 验证邮箱是否有效
     */
    public static boolean isValidEmail(String email) {
        if (email == null || email.isEmpty()) {
            return false;
        }
        return EMAIL_PATTERN.matcher(email).matches();
    }

    /**
     * 验证昵称是否有效
     */
    public static boolean isValidNickname(String nickname) {
        if (nickname == null || nickname.isEmpty()) {
            return false;
        }
        return NICKNAME_PATTERN.matcher(nickname).matches();
    }

    /**
     * 获取用户名验证规则说明
     */
    public static String getUsernameRule() {
        return "用户名必须是4-20位字母、数字或下划线";
    }

    /**
     * 获取密码验证规则说明
     */
    public static String getPasswordRule() {
        return "密码必须是8-20位，至少包含字母和数字";
    }

    /**
     * 获取邮箱验证规则说明
     */
    public static String getEmailRule() {
        return "请输入有效的邮箱地址";
    }

    /**
     * 获取昵称验证规则说明
     */
    public static String getNicknameRule() {
        return "昵称必须是2-20位中文、字母、数字或下划线";
    }
}