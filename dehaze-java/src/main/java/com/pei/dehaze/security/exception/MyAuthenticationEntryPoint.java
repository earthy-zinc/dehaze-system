package com.pei.dehaze.security.exception;

import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.common.util.ResponseUtils;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.security.authentication.BadCredentialsException;
import org.springframework.security.core.AuthenticationException;
import org.springframework.security.web.AuthenticationEntryPoint;
import org.springframework.stereotype.Component;

/**
 * 认证异常处理
 *
 * @author earthyzinc
 * @since 2.0.0
 */
@Component
public class MyAuthenticationEntryPoint implements AuthenticationEntryPoint {
    @Override
    public void commence(HttpServletRequest request, HttpServletResponse response,
                         AuthenticationException authException) {
        int status = response.getStatus();
        if (status == HttpServletResponse.SC_NOT_FOUND) {
            ResponseUtils.writeErrMsg(response, ResultCode.RESOURCE_NOT_FOUND);
        } else if (authException instanceof BadCredentialsException) {
            ResponseUtils.writeErrMsg(response, ResultCode.USERNAME_OR_PASSWORD_ERROR);
        } else {
            ResponseUtils.writeErrMsg(response, ResultCode.TOKEN_INVALID);
        }
    }
}
