package com.pei.dehaze.repository;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.verify;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.api.AlgorithmAPI;
import com.pei.dehaze.sdk.model.algorithm.Algorithm;
import com.pei.dehaze.sdk.model.algorithm.AlgorithmQuery;

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
public class AlgorithmRepositoryTest {

    private AlgorithmRepository algorithmRepository;

    @Mock
    private AlgorithmRepository.AlgorithmCallback algorithmCallback;

    @Mock
    private AlgorithmRepository.AlgorithmDetailCallback algorithmDetailCallback;

    @Before
    public void setUp() {
        algorithmRepository = new AlgorithmRepository();
    }

    @Test
    public void testGetAlgorithmsSuccess() {
        // 模拟 AlgorithmAPI.getList 成功响应
        List<Algorithm> mockAlgorithms = new ArrayList<>();
        Algorithm algorithm1 = new Algorithm();
        algorithm1.setId(1);
        algorithm1.setName("算法1");
        algorithm1.setType("类型1");
        algorithm1.setDescription("描述1");
        mockAlgorithms.add(algorithm1);

        try (MockedStatic<AlgorithmAPI> mockedAlgorithmAPI = Mockito.mockStatic(AlgorithmAPI.class)) {
            mockedAlgorithmAPI.when(() -> AlgorithmAPI.getList(any(AlgorithmQuery.class), any()))
                    .thenAnswer(invocation -> {
                        ApiCallback<List<Algorithm>> callback = invocation.getArgument(1);
                        callback.onSuccess(mockAlgorithms);
                        return null;
                    });

            // 执行获取算法列表
            AlgorithmQuery query = new AlgorithmQuery();
            algorithmRepository.getAlgorithms(query, algorithmCallback);

            // 验证回调被调用
            verify(algorithmCallback).onSuccess(mockAlgorithms);
        }
    }

    @Test
    public void testGetAlgorithmsError() {
        // 模拟 AlgorithmAPI.getList 错误响应
        try (MockedStatic<AlgorithmAPI> mockedAlgorithmAPI = Mockito.mockStatic(AlgorithmAPI.class)) {
            mockedAlgorithmAPI.when(() -> AlgorithmAPI.getList(any(AlgorithmQuery.class), any()))
                    .thenAnswer(invocation -> {
                        ApiCallback<List<Algorithm>> callback = invocation.getArgument(1);
                        callback.onError(404, "Not Found");
                        return null;
                    });

            // 执行获取算法列表
            AlgorithmQuery query = new AlgorithmQuery();
            algorithmRepository.getAlgorithms(query, algorithmCallback);

            // 验证回调被调用
            verify(algorithmCallback).onError("Error 404: Not Found");
        }
    }

    @Test
    public void testGetAlgorithmsNetworkFailure() {
        // 模拟 AlgorithmAPI.getList 网络失败
        try (MockedStatic<AlgorithmAPI> mockedAlgorithmAPI = Mockito.mockStatic(AlgorithmAPI.class)) {
            mockedAlgorithmAPI.when(() -> AlgorithmAPI.getList(any(AlgorithmQuery.class), any()))
                    .thenAnswer(invocation -> {
                        ApiCallback<List<Algorithm>> callback = invocation.getArgument(1);
                        callback.onFailure(new com.pei.dehaze.sdk.network.ApiException(0, "Network error"));
                        return null;
                    });

            // 执行获取算法列表
            AlgorithmQuery query = new AlgorithmQuery();
            algorithmRepository.getAlgorithms(query, algorithmCallback);

            // 验证回调被调用
            verify(algorithmCallback).onError("Network error: Network error");
        }
    }

    @Test
    public void testGetAlgorithmDetailSuccess() {
        // 模拟 AlgorithmAPI.getAlgorithmInfoById 成功响应
        Algorithm mockAlgorithm = new Algorithm();
        mockAlgorithm.setId(1);
        mockAlgorithm.setName("算法1");
        mockAlgorithm.setType("类型1");
        mockAlgorithm.setDescription("描述1");

        try (MockedStatic<AlgorithmAPI> mockedAlgorithmAPI = Mockito.mockStatic(AlgorithmAPI.class)) {
            mockedAlgorithmAPI.when(() -> AlgorithmAPI.getAlgorithmInfoById(any(Integer.class), any()))
                    .thenAnswer(invocation -> {
                        ApiCallback<Algorithm> callback = invocation.getArgument(1);
                        callback.onSuccess(mockAlgorithm);
                        return null;
                    });

            // 执行获取算法详情
            algorithmRepository.getAlgorithmDetail(1, algorithmDetailCallback);

            // 验证回调被调用
            verify(algorithmDetailCallback).onSuccess(mockAlgorithm);
        }
    }

    @Test
    public void testGetAlgorithmDetailError() {
        // 模拟 AlgorithmAPI.getAlgorithmInfoById 错误响应
        try (MockedStatic<AlgorithmAPI> mockedAlgorithmAPI = Mockito.mockStatic(AlgorithmAPI.class)) {
            mockedAlgorithmAPI.when(() -> AlgorithmAPI.getAlgorithmInfoById(any(Integer.class), any()))
                    .thenAnswer(invocation -> {
                        ApiCallback<Algorithm> callback = invocation.getArgument(1);
                        callback.onError(404, "Not Found");
                        return null;
                    });

            // 执行获取算法详情
            algorithmRepository.getAlgorithmDetail(1, algorithmDetailCallback);

            // 验证回调被调用
            verify(algorithmDetailCallback).onError("Error 404: Not Found");
        }
    }

    @Test
    public void testGetAlgorithmDetailNetworkFailure() {
        // 模拟 AlgorithmAPI.getAlgorithmInfoById 网络失败
        try (MockedStatic<AlgorithmAPI> mockedAlgorithmAPI = Mockito.mockStatic(AlgorithmAPI.class)) {
            mockedAlgorithmAPI.when(() -> AlgorithmAPI.getAlgorithmInfoById(any(Integer.class), any()))
                    .thenAnswer(invocation -> {
                        ApiCallback<Algorithm> callback = invocation.getArgument(1);
                        callback.onFailure(new com.pei.dehaze.sdk.network.ApiException(0, "Network error"));
                        return null;
                    });

            // 执行获取算法详情
            algorithmRepository.getAlgorithmDetail(1, algorithmDetailCallback);

            // 验证回调被调用
            verify(algorithmDetailCallback).onError("Network error: Network error");
        }
    }
}