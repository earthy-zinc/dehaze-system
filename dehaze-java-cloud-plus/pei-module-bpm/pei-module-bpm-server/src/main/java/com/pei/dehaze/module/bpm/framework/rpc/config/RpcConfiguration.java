package com.pei.dehaze.module.bpm.framework.rpc.config;

import com.pei.dehaze.module.system.api.dept.DeptApi;
import com.pei.dehaze.module.system.api.dept.PostApi;
import com.pei.dehaze.module.system.api.dict.DictDataApi;
import com.pei.dehaze.module.system.api.permission.PermissionApi;
import com.pei.dehaze.module.system.api.permission.RoleApi;
import com.pei.dehaze.module.system.api.sms.SmsSendApi;
import com.pei.dehaze.module.system.api.user.AdminUserApi;
import org.springframework.cloud.openfeign.EnableFeignClients;
import org.springframework.context.annotation.Configuration;

@Configuration(value = "bpmRpcConfiguration", proxyBeanMethods = false)
@EnableFeignClients(clients = {RoleApi.class, DeptApi.class, PostApi.class, AdminUserApi.class, SmsSendApi.class, DictDataApi.class,
        PermissionApi.class})
public class RpcConfiguration {
}
