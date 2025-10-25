package com.pei.dehaze.repository;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.verify;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.api.UserAPI;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.user.UserPageVO;
import com.pei.dehaze.sdk.model.user.UserQuery;

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
public class UserRepositoryTest {

    private UserRepository userRepository;

    @Mock
    private UserRepository.UserCallback userCallback;

    @Before
    public void setUp() {
        userRepository = new UserRepository();
    }

    @Test
    public void testGetUsersSuccess() {
        // 模拟 UserAPI.getPage 成功响应
        List<UserPageVO> mockUsers = new ArrayList<>();
        UserPageVO user1 = new UserPageVO();
        user1.setId(1);
        user1.setUsername("user1");
        user1.setNickname("User One");
        user1.setMobile("13800138000");
        mockUsers.add(user1);

        PageResult<UserPageVO> mockResult = new PageResult<>();
        mockResult.setList(mockUsers);
        mockResult.setTotal(1);

        try (MockedStatic<UserAPI> mockedUserAPI = Mockito.mockStatic(UserAPI.class)) {
            mockedUserAPI.when(() -> UserAPI.getPage(any(UserQuery.class), any()))
                    .thenAnswer(invocation -> {
                        ApiCallback<PageResult<UserPageVO>> callback = invocation.getArgument(1);
                        callback.onSuccess(mockResult);
                        return null;
                    });

            // 执行获取用户
            userRepository.getUsers(userCallback);

            // 验证回调被调用
            verify(userCallback).onSuccess(mockUsers);
        }
    }

    @Test
    public void testGetUsersError() {
        // 模拟 UserAPI.getPage 错误响应
        try (MockedStatic<UserAPI> mockedUserAPI = Mockito.mockStatic(UserAPI.class)) {
            mockedUserAPI.when(() -> UserAPI.getPage(any(UserQuery.class), any()))
                    .thenAnswer(invocation -> {
                        ApiCallback<PageResult<UserPageVO>> callback = invocation.getArgument(1);
                        callback.onError(404, "Not Found");
                        return null;
                    });

            // 执行获取用户
            userRepository.getUsers(userCallback);

            // 验证回调被调用
            verify(userCallback).onError("Error 404: Not Found");
        }
    }

    @Test
    public void testGetUsersNetworkFailure() {
        // 模拟 UserAPI.getPage 网络失败
        try (MockedStatic<UserAPI> mockedUserAPI = Mockito.mockStatic(UserAPI.class)) {
            mockedUserAPI.when(() -> UserAPI.getPage(any(UserQuery.class), any()))
                    .thenAnswer(invocation -> {
                        ApiCallback<PageResult<UserPageVO>> callback = invocation.getArgument(1);
                        callback.onFailure(new com.pei.dehaze.sdk.network.ApiException(0, "Network error"));
                        return null;
                    });

            // 执行获取用户
            userRepository.getUsers(userCallback);

            // 验证回调被调用
            verify(userCallback).onError("Network error: Network error");
        }
    }
}