package com.pei.dehaze.ui.algorithm.viewmodel;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.verify;

import androidx.arch.core.executor.testing.InstantTaskExecutorRule;
import androidx.lifecycle.Observer;

import com.pei.dehaze.repository.AlgorithmRepository;
import com.pei.dehaze.sdk.model.algorithm.Algorithm;
import com.pei.dehaze.sdk.model.algorithm.AlgorithmQuery;

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
public class AlgorithmViewModelTest {

    // 确保 LiveData 在测试中立即执行
    @Rule
    public InstantTaskExecutorRule instantExecutorRule = new InstantTaskExecutorRule();

    private AlgorithmViewModel algorithmViewModel;

    @Mock
    private AlgorithmRepository algorithmRepository;

    @Mock
    private Observer<List<Algorithm>> algorithmListObserver;

    @Mock
    private Observer<Algorithm> algorithmDetailObserver;

    @Mock
    private Observer<Boolean> loadingObserver;

    @Mock
    private Observer<String> errorObserver;

    @Before
    public void setUp() {
        MockitoAnnotations.initMocks(this);
        algorithmViewModel = new AlgorithmViewModel();
        // 使用反射注入 mock 的 repository
        try {
            java.lang.reflect.Field field = AlgorithmViewModel.class.getDeclaredField("algorithmRepository");
            field.setAccessible(true);
            field.set(algorithmViewModel, algorithmRepository);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Test
    public void testInitialState() {
        // 测试初始状态
        assertNotNull(algorithmViewModel.getAlgorithmList());
        assertNotNull(algorithmViewModel.getAlgorithmDetail());
        assertNotNull(algorithmViewModel.getLoading());
        assertNotNull(algorithmViewModel.getError());

        assertNull(algorithmViewModel.getLoading().getValue());
        assertNull(algorithmViewModel.getError().getValue());
    }

    @Test
    public void testLoadAlgorithmsSuccess() {
        // 模拟成功获取算法列表
        List<Algorithm> mockAlgorithms = new ArrayList<>();
        Algorithm algorithm1 = new Algorithm();
        algorithm1.setId(1);
        algorithm1.setName("算法1");
        algorithm1.setType("类型1");
        algorithm1.setDescription("描述1");
        mockAlgorithms.add(algorithm1);

        doAnswer(invocation -> {
            AlgorithmRepository.AlgorithmCallback callback = invocation.getArgument(1);
            callback.onSuccess(mockAlgorithms);
            return null;
        }).when(algorithmRepository).getAlgorithms(any(AlgorithmQuery.class), any());

        // 观察数据变化
        algorithmViewModel.getAlgorithmList().observeForever(algorithmListObserver);
        algorithmViewModel.getLoading().observeForever(loadingObserver);

        // 执行加载算法
        AlgorithmQuery query = new AlgorithmQuery();
        algorithmViewModel.loadAlgorithms(query);

        // 验证状态变化
        verify(loadingObserver).onChanged(true);
        verify(loadingObserver).onChanged(false);
        verify(algorithmListObserver).onChanged(mockAlgorithms);
    }

    @Test
    public void testLoadAlgorithmsError() {
        // 模拟获取算法列表失败
        String errorMessage = "Network error";

        doAnswer(invocation -> {
            AlgorithmRepository.AlgorithmCallback callback = invocation.getArgument(1);
            callback.onError(errorMessage);
            return null;
        }).when(algorithmRepository).getAlgorithms(any(AlgorithmQuery.class), any());

        // 观察数据变化
        algorithmViewModel.getError().observeForever(errorObserver);
        algorithmViewModel.getLoading().observeForever(loadingObserver);

        // 执行加载算法
        AlgorithmQuery query = new AlgorithmQuery();
        algorithmViewModel.loadAlgorithms(query);

        // 验证状态变化
        verify(loadingObserver).onChanged(true);
        verify(loadingObserver).onChanged(false);
        verify(errorObserver).onChanged(errorMessage);
    }

    @Test
    public void testLoadAlgorithmDetailSuccess() {
        // 模拟成功获取算法详情
        Algorithm mockAlgorithm = new Algorithm();
        mockAlgorithm.setId(1);
        mockAlgorithm.setName("算法1");
        mockAlgorithm.setType("类型1");
        mockAlgorithm.setDescription("描述1");

        doAnswer(invocation -> {
            AlgorithmRepository.AlgorithmDetailCallback callback = invocation.getArgument(1);
            callback.onSuccess(mockAlgorithm);
            return null;
        }).when(algorithmRepository).getAlgorithmDetail(any(Integer.class), any());

        // 观察数据变化
        algorithmViewModel.getAlgorithmDetail().observeForever(algorithmDetailObserver);
        algorithmViewModel.getLoading().observeForever(loadingObserver);

        // 执行加载算法详情
        algorithmViewModel.loadAlgorithmDetail(1);

        // 验证状态变化
        verify(loadingObserver).onChanged(true);
        verify(loadingObserver).onChanged(false);
        verify(algorithmDetailObserver).onChanged(mockAlgorithm);
    }

    @Test
    public void testLoadAlgorithmDetailError() {
        // 模拟获取算法详情失败
        String errorMessage = "Network error";

        doAnswer(invocation -> {
            AlgorithmRepository.AlgorithmDetailCallback callback = invocation.getArgument(1);
            callback.onError(errorMessage);
            return null;
        }).when(algorithmRepository).getAlgorithmDetail(any(Integer.class), any());

        // 观察数据变化
        algorithmViewModel.getError().observeForever(errorObserver);
        algorithmViewModel.getLoading().observeForever(loadingObserver);

        // 执行加载算法详情
        algorithmViewModel.loadAlgorithmDetail(1);

        // 验证状态变化
        verify(loadingObserver).onChanged(true);
        verify(loadingObserver).onChanged(false);
        verify(errorObserver).onChanged(errorMessage);
    }
}