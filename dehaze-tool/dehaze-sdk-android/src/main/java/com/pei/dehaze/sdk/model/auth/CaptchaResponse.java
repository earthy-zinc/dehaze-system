package com.pei.dehaze.sdk.model.auth;

import lombok.Data;

/**
 * 验证码响应模型类
 * 用于封装获取验证码接口的响应数据
 */
@Data
public class CaptchaResponse {
    private String captchaKey; // 验证码唯一标识
    private String captchaBase64; // 验证码图片的Base64编码
}