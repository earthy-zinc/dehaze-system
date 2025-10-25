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

import com.pei.dehaze.repository.DeptRepository;
import com.pei.dehaze.sdk.model.dept.DeptVO;

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
public class DeptViewModelTest {

    // 确保 LiveData 在测试中立即执行
    @Rule
    public InstantTaskExecutorRule instantExecutorRule = new InstantTaskExecutorRule();

    private DeptViewModel deptViewModel;

    @Mock
    private DeptRepository deptRepository;

    @Mock
    private Observer<List<DeptVO>> deptListObserver;

    @Mock
    private Observer<Boolean> loadingObserver;

    @Mock
    private Observer<String> errorObserver;

    @Before
    public void setUp() {
        MockitoAnnotations.initMocks(this);
        deptViewModel = new DeptViewModel();
        // 使用反射注入 mock 的 repository
        try {
            java.lang.reflect.Field field = DeptViewModel.class.getDeclaredField("deptRepository");
            field.setAccessible(true);
            field.set(deptViewModel, deptRepository);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Test
    public void testInitialState() {
        // 测试初始状态
        assertNotNull(deptViewModel.getDeptList());
        assertNotNull(deptViewModel.getLoading());
        assertNotNull(deptViewModel.getError());

        assertNull(deptViewModel.getLoading().getValue());
        assertNull(deptViewModel.getError().getValue());
    }

    @Test
    public void testLoadDeptsSuccess() {
        // 模拟成功获取部门列表
        List<DeptVO> mockDepts = new ArrayList<>();
        DeptVO dept1 = new DeptVO();
        dept1.setId(1);
        dept1.setName("部门1");
        dept1.setStatus(1);
        mockDepts.add(dept1);

        doAnswer(invocation -> {
            DeptRepository.DeptCallback callback = invocation.getArgument(0);
            callback.onSuccess(mockDepts);
            return null;
        }).when(deptRepository).getDepts(any());

        // 观察数据变化
        deptViewModel.getDeptList().observeForever(deptListObserver);
        deptViewModel.getLoading().observeForever(loadingObserver);

        // 执行加载部门
        deptViewModel.loadDepts();

        // 验证状态变化
        verify(loadingObserver).onChanged(true);
        verify(loadingObserver).onChanged(false);
        verify(deptListObserver).onChanged(mockDepts);
    }

    @Test
    public void testLoadDeptsError() {
        // 模拟获取部门列表失败
        String errorMessage = "Network error";

        doAnswer(invocation -> {
            DeptRepository.DeptCallback callback = invocation.getArgument(0);
            callback.onError(errorMessage);
            return null;
        }).when(deptRepository).getDepts(any());

        // 观察数据变化
        deptViewModel.getError().observeForever(errorObserver);
        deptViewModel.getLoading().observeForever(loadingObserver);

        // 执行加载部门
        deptViewModel.loadDepts();

        // 验证状态变化
        verify(loadingObserver).onChanged(true);
        verify(loadingObserver).onChanged(false);
        verify(errorObserver).onChanged(errorMessage);
    }
}