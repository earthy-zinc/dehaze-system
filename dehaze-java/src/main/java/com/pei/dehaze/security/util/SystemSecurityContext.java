package com.pei.dehaze.security.util;

import com.pei.dehaze.common.constant.SystemConstants;
import com.pei.dehaze.common.enums.DataScopeEnum;
import com.pei.dehaze.security.model.SysUserDetails;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;

import java.util.Collections;

/**
 * 系统安全上下文工具
 * 用于异步任务、MQ消费者、定时任务等无HTTP上下文场景，注入系统用户身份
 *
 * @author earthyzinc
 * @since 1.0.0
 */
public class SystemSecurityContext {

    /**
     * 设置系统用户安全上下文
     */
    public static void setSystemContext() {
        SysUserDetails systemUser = new SysUserDetails();
        systemUser.setUserId(SystemConstants.SYSTEM_USER_ID);
        systemUser.setUsername(SystemConstants.SYSTEM_USERNAME);
        systemUser.setNickname(SystemConstants.SYSTEM_USERNAME);
        systemUser.setDeptId(SystemConstants.SYSTEM_USER_ID);
        systemUser.setDataScope(DataScopeEnum.ALL.getValue());
        systemUser.setEnabled(true);
        systemUser.setAuthorities(Collections.emptySet());
        systemUser.setPerms(Collections.emptySet());

        UsernamePasswordAuthenticationToken auth =
                new UsernamePasswordAuthenticationToken(systemUser, null, Collections.emptyList());
        SecurityContextHolder.getContext().setAuthentication(auth);
    }

    /**
     * 清除安全上下文
     */
    public static void clearContext() {
        SecurityContextHolder.clearContext();
    }
}
