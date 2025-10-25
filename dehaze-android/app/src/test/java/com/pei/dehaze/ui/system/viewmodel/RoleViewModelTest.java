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

import com.pei.dehaze.repository.RoleRepository;
import com.pei.dehaze.sdk.model.role.RolePageVO;

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
public class RoleViewModelTest {

    // 确保 LiveData 在测试中立即执行
    @Rule
    public InstantTaskExecutorRule instantExecutorRule = new InstantTaskExecutorRule();

    private RoleViewModel roleViewModel;

    @Mock
    private RoleRepository roleRepository;

    @Mock
    private Observer<List<RolePageVO>> roleListObserver;

    @Mock
    private Observer<Boolean> loadingObserver;

    @Mock
    private Observer<String> errorObserver;

    @Before
    public void setUp() {
        MockitoAnnotations.initMocks(this);
        roleViewModel = new RoleViewModel();
        // 使用反射注入 mock 的 repository
        try {
            java.lang.reflect.Field field = RoleViewModel.class.getDeclaredField("roleRepository");
            field.setAccessible(true);
            field.set(roleViewModel, roleRepository);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Test
    public void testInitialState() {
        // 测试初始状态
        assertNotNull(roleViewModel.getRoleList());
        assertNotNull(roleViewModel.getLoading());
        assertNotNull(roleViewModel.getError());

        assertNull(roleViewModel.getLoading().getValue());
        assertNull(roleViewModel.getError().getValue());
    }

    @Test
    public void testLoadRolesSuccess() {
        // 模拟成功获取角色列表
        List<RolePageVO> mockRoles = new ArrayList<>();
        RolePageVO role1 = new RolePageVO();
        role1.setId(1);
        role1.setName("角色1");
        role1.setCode("ROLE_1");
        mockRoles.add(role1);

        doAnswer(invocation -> {
            RoleRepository.RoleCallback callback = invocation.getArgument(0);
            callback.onSuccess(mockRoles);
            return null;
        }).when(roleRepository).getRoles(any());

        // 观察数据变化
        roleViewModel.getRoleList().observeForever(roleListObserver);
        roleViewModel.getLoading().observeForever(loadingObserver);

        // 执行加载角色
        roleViewModel.loadRoles();

        // 验证状态变化
        verify(loadingObserver).onChanged(true);
        verify(loadingObserver).onChanged(false);
        verify(roleListObserver).onChanged(mockRoles);
    }

    @Test
    public void testLoadRolesError() {
        // 模拟获取角色列表失败
        String errorMessage = "Network error";

        doAnswer(invocation -> {
            RoleRepository.RoleCallback callback = invocation.getArgument(0);
            callback.onError(errorMessage);
            return null;
        }).when(roleRepository).getRoles(any());

        // 观察数据变化
        roleViewModel.getError().observeForever(errorObserver);
        roleViewModel.getLoading().observeForever(loadingObserver);

        // 执行加载角色
        roleViewModel.loadRoles();

        // 验证状态变化
        verify(loadingObserver).onChanged(true);
        verify(loadingObserver).onChanged(false);
        verify(errorObserver).onChanged(errorMessage);
    }
}