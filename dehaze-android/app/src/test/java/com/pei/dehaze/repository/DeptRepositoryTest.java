package com.pei.dehaze.repository;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.verify;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.api.DeptAPI;
import com.pei.dehaze.sdk.model.dept.DeptQuery;
import com.pei.dehaze.sdk.model.dept.DeptVO;

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
public class DeptRepositoryTest {

    private DeptRepository deptRepository;

    @Mock
    private DeptRepository.DeptCallback deptCallback;

    @Before
    public void setUp() {
        deptRepository = new DeptRepository();
    }

    @Test
    public void testGetDeptsSuccess() {
        // 模拟 DeptAPI.getList 成功响应
        List<DeptVO> mockDepts = new ArrayList<>();
        DeptVO dept1 = new DeptVO();
        dept1.setId(1);
        dept1.setName("部门1");
        dept1.setStatus(1);
        mockDepts.add(dept1);

        try (MockedStatic<DeptAPI> mockedDeptAPI = Mockito.mockStatic(DeptAPI.class)) {
            mockedDeptAPI.when(() -> DeptAPI.getList(any(DeptQuery.class), any()))
                    .thenAnswer(invocation -> {
                        ApiCallback<List<DeptVO>> callback = invocation.getArgument(1);
                        callback.onSuccess(mockDepts);
                        return null;
                    });

            // 执行获取部门
            deptRepository.getDepts(deptCallback);

            // 验证回调被调用
            verify(deptCallback).onSuccess(mockDepts);
        }
    }

    @Test
    public void testGetDeptsError() {
        // 模拟 DeptAPI.getList 错误响应
        try (MockedStatic<DeptAPI> mockedDeptAPI = Mockito.mockStatic(DeptAPI.class)) {
            mockedDeptAPI.when(() -> DeptAPI.getList(any(DeptQuery.class), any()))
                    .thenAnswer(invocation -> {
                        ApiCallback<List<DeptVO>> callback = invocation.getArgument(1);
                        callback.onError(404, "Not Found");
                        return null;
                    });

            // 执行获取部门
            deptRepository.getDepts(deptCallback);

            // 验证回调被调用
            verify(deptCallback).onError("Error 404: Not Found");
        }
    }

    @Test
    public void testGetDeptsNetworkFailure() {
        // 模拟 DeptAPI.getList 网络失败
        try (MockedStatic<DeptAPI> mockedDeptAPI = Mockito.mockStatic(DeptAPI.class)) {
            mockedDeptAPI.when(() -> DeptAPI.getList(any(DeptQuery.class), any()))
                    .thenAnswer(invocation -> {
                        ApiCallback<List<DeptVO>> callback = invocation.getArgument(1);
                        callback.onFailure(new com.pei.dehaze.sdk.network.ApiException(0, "Network error"));
                        return null;
                    });

            // 执行获取部门
            deptRepository.getDepts(deptCallback);

            // 验证回调被调用
            verify(deptCallback).onError("Network error: Network error");
        }
    }
}