package com.pei.dehaze.ui.system.viewmodel;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;

import androidx.arch.core.executor.testing.InstantTaskExecutorRule;
import androidx.lifecycle.Observer;

import com.pei.dehaze.repository.UserRepository;
import com.pei.dehaze.sdk.model.user.UserPageVO;

import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.robolectric.RobolectricTestRunner;
import org.robolectric.annotation.Config;

import java.util.ArrayList;
import java.util.List;

@Config(sdk = 28)
@RunWith(RobolectricTestRunner.class)
public class UserViewModelTest {

    // 确保 LiveData 在测试中立即执行
    @Rule
    public InstantTaskExecutorRule instantExecutorRule = new InstantTaskExecutorRule();

    private UserViewModel userViewModel;

    @Mock
    private UserRepository userRepository;

    @Mock
    private Observer<List<UserPageVO>> userListObserver;

    @Mock
    private Observer<Boolean> loadingObserver;

    @Mock
    private Observer<String> errorObserver;

    @Before
    public void setUp() {
        MockitoAnnotations.initMocks(this);
        userViewModel = new UserViewModel();
        // 使用反射注入 mock 的 repository
        try {
            java.lang.reflect.Field field = UserViewModel.class.getDeclaredField("userRepository");
            field.setAccessible(true);
            field.set(userViewModel, userRepository);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Test
    public void testInitialState() {
        // 测试初始状态
        assertNotNull(userViewModel.getUserList());
        assertNotNull(userViewModel.getLoading());
        assertNotNull(userViewModel.getError());

        assertNull(userViewModel.getLoading().getValue());
        assertNull(userViewModel.getError().getValue());
    }

    @Test
    public void testLoadUsersSuccess() {
        // 模拟成功获取用户列表
        List<UserPageVO> mockUsers = new ArrayList<>();
        UserPageVO user1 = new UserPageVO();
        user1.setId(1);
        user1.setUsername("user1");
        user1.setNickname("User One");
        user1.setMobile("13800138000");
        mockUsers.add(user1);

        doAnswer(invocation -> {
            UserRepository.UserCallback callback = invocation.getArgument(0);
            callback.onSuccess(mockUsers);
            return null;
        }).when(userRepository).getUsers(any());

        // 观察数据变化
        userViewModel.getUserList().observeForever(userListObserver);
        userViewModel.getLoading().observeForever(loadingObserver);

        // 执行加载用户
        userViewModel.loadUsers();

        // 验证状态变化
        verify(loadingObserver).onChanged(true);
        verify(loadingObserver).onChanged(false);
        verify(userListObserver).onChanged(mockUsers);
    }

    @Test
    public void testLoadUsersError() {
        // 模拟获取用户列表失败
        String errorMessage = "Network error";

        doAnswer(invocation -> {
            UserRepository.UserCallback callback = invocation.getArgument(0);
            callback.onError(errorMessage);
            return null;
        }).when(userRepository).getUsers(any());

        // 观察数据变化
        userViewModel.getError().observeForever(errorObserver);
        userViewModel.getLoading().observeForever(loadingObserver);

        // 执行加载用户
        userViewModel.loadUsers();

        // 验证状态变化
        verify(loadingObserver).onChanged(true);
        verify(loadingObserver).onChanged(false);
        verify(errorObserver).onChanged(errorMessage);
    }
}