package com.pei.dehaze.service;

import com.pei.dehaze.model.dto.CaptchaResult;
import com.pei.dehaze.model.form.LoginForm;
import com.pei.dehaze.model.dto.LoginResult;
import com.pei.dehaze.model.form.RegisterForm;

public interface AuthService {

    LoginResult login(LoginForm form);

    LoginResult register(RegisterForm form);

    void logout();

    CaptchaResult getCaptcha();
}
