package com.pei.dehaze.repository;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.verify;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.api.RoleAPI;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.role.RolePageVO;
import com.pei.dehaze.sdk.model.role.RoleQuery;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.mockito.junit.MockitoJUnitRunner;

import java.util.ArrayList;
import java.util.List;

@RunWith(MockitoJUnitRunner.class)
public class RoleRepositoryTest {

    private RoleRepository roleRepository;

    @Mock
    private RoleRepository.RoleCallback roleCallback;

    @Before
    public void setUp() {
        roleRepository = new RoleRepository();
    }

    @Test
    public void testGetRolesSuccess() {
        // 模拟 RoleAPI.getPage 成功响应
        List<RolePageVO> mockRoles = new ArrayList<>();
        RolePageVO role1 = new RolePageVO();
        role1.setId(1);
        role1.setName("角色1");
        role1.setCode("ROLE_1");
        mockRoles.add(role1);

        PageResult<RolePageVO> mockResult = new PageResult<>();
        mockResult.setList(mockRoles);
        mockResult.setTotal(1);

        try (MockedStatic<RoleAPI> mockedRoleAPI = Mockito.mockStatic(RoleAPI.class)) {
            mockedRoleAPI.when(() -> RoleAPI.getPage(any(RoleQuery.class), any()))
                    .thenAnswer(invocation -> {
                        ApiCallback<PageResult<RolePageVO>> callback = invocation.getArgument(1);
                        callback.onSuccess(mockResult);
                        return null;
                    });

            // 执行获取角色
            roleRepository.getRoles(roleCallback);

            // 验证回调被调用
            verify(roleCallback).onSuccess(mockRoles);
        }
    }

    @Test
    public void testGetRolesError() {
        // 模拟 RoleAPI.getPage 错误响应
        try (MockedStatic<RoleAPI> mockedRoleAPI = Mockito.mockStatic(RoleAPI.class)) {
            mockedRoleAPI.when(() -> RoleAPI.getPage(any(RoleQuery.class), any()))
                    .thenAnswer(invocation -> {
                        ApiCallback<PageResult<RolePageVO>> callback = invocation.getArgument(1);
                        callback.onError(404, "Not Found");
                        return null;
                    });

            // 执行获取角色
            roleRepository.getRoles(roleCallback);

            // 验证回调被调用
            verify(roleCallback).onError("Error 404: Not Found");
        }
    }

    @Test
    public void testGetRolesNetworkFailure() {
        // 模拟 RoleAPI.getPage 网络失败
        try (MockedStatic<RoleAPI> mockedRoleAPI = Mockito.mockStatic(RoleAPI.class)) {
            mockedRoleAPI.when(() -> RoleAPI.getPage(any(RoleQuery.class), any()))
                    .thenAnswer(invocation -> {
                        ApiCallback<PageResult<RolePageVO>> callback = invocation.getArgument(1);
                        callback.onFailure(new com.pei.dehaze.sdk.network.ApiException(0, "Network error"));
                        return null;
                    });

            // 执行获取角色
            roleRepository.getRoles(roleCallback);

            // 验证回调被调用
            verify(roleCallback).onError("Network error: Network error");
        }
    }
}