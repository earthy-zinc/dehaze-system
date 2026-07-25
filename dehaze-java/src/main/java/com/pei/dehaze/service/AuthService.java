package com.pei.dehaze.service;

import com.pei.dehaze.model.dto.CaptchaResult;
import com.pei.dehaze.model.dto.LoginForm;
import com.pei.dehaze.model.dto.LoginResult;
import com.pei.dehaze.model.dto.RegisterForm;

public interface AuthService {

    LoginResult login(LoginForm form);

    LoginResult register(RegisterForm form);

    void logout();

    CaptchaResult getCaptcha();
}
