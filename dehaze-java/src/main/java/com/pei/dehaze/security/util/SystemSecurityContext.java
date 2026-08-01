package com.pei.dehaze.security.util;

import com.pei.dehaze.common.constant.SystemConstants;
import com.pei.dehaze.common.enums.DataScopeEnum;
import com.pei.dehaze.filter.TraceIdFilter;
import com.pei.dehaze.security.model.SysUserDetails;
import org.slf4j.MDC;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;

import java.util.Collections;
import java.util.UUID;

/**
 * 系统安全上下文工具
 * 用于异步任务、MQ消费者、定时任务等无HTTP上下文场景，注入系统用户身份与 trace_id
 *
 * @author earthyzinc
 * @since 1.0.0
 */
public class SystemSecurityContext {

    /**
     * 设置系统用户安全上下文（含任务级 trace_id）
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

        // 后台任务注入任务级 trace_id
        String traceId = UUID.randomUUID().toString().replace("-", "");
        MDC.put(TraceIdFilter.MDC_TRACE_ID, traceId);
    }

    /**
     * 清除安全上下文（含 MDC）
     */
    public static void clearContext() {
        SecurityContextHolder.clearContext();
        MDC.remove(TraceIdFilter.MDC_TRACE_ID);
        MDC.remove(TraceIdFilter.MDC_METHOD);
        MDC.remove(TraceIdFilter.MDC_PATH);
        MDC.remove(TraceIdFilter.MDC_IP);
        MDC.remove(TraceIdFilter.MDC_USER_AGENT);
        MDC.remove("user_id");
    }
}
