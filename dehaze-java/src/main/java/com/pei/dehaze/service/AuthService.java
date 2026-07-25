package com.pei.dehaze.service;

import com.pei.dehaze.model.dto.CaptchaResult;
import com.pei.dehaze.model.dto.LoginForm;
import com.pei.dehaze.model.dto.LoginResult;

public interface AuthService {

    LoginResult login(LoginForm form);

    void logout();

    CaptchaResult getCaptcha();

    LoginResult refreshToken(String refreshToken);
}
