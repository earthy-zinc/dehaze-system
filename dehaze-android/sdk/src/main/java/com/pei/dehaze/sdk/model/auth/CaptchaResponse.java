package com.pei.dehaze.sdk.model.auth;

import lombok.Data;

/**
 * 验证码响应模型
 * 对齐后端 CaptchaData：captchaKey、captchaBase64
 */
@Data
public class CaptchaResponse {
    /** 验证码唯一标识 */
    private String captchaKey;
    /** 验证码图片的 Base64 编码 */
    private String captchaBase64;
}
