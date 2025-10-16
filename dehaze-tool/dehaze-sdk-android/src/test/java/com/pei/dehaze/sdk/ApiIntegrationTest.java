package com.pei.dehaze.sdk;

import com.pei.dehaze.sdk.api.*;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.algorithm.Algorithm;
import com.pei.dehaze.sdk.model.algorithm.AlgorithmQuery;
import com.pei.dehaze.sdk.model.dataset.Dataset;
import com.pei.dehaze.sdk.model.dataset.DatasetQuery;
import com.pei.dehaze.sdk.model.dept.DeptQuery;
import com.pei.dehaze.sdk.model.dept.DeptVO;
import com.pei.dehaze.sdk.model.dict.DictTypePageVO;
import com.pei.dehaze.sdk.model.dict.DictTypeQuery;
import com.pei.dehaze.sdk.model.menu.MenuQuery;
import com.pei.dehaze.sdk.model.menu.MenuVO;
import com.pei.dehaze.sdk.model.role.RolePageVO;
import com.pei.dehaze.sdk.model.role.RoleQuery;
import com.pei.dehaze.sdk.model.user.UserInfo;
import com.pei.dehaze.sdk.model.user.UserPageVO;
import com.pei.dehaze.sdk.model.user.UserQuery;
import org.junit.Before;
import org.junit.Test;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

import java.util.List;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.fail;

/**
 * API接口集成测试示例
 * 展示如何测试各个模块的API接口
 */
public class ApiIntegrationTest {

    @Mock
    private DehazeSDK dehazeSDK;

    @Before
    public void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testUserAPI() {
        // 测试用户相关API
        try {
            // 测试获取用户信息
            UserAPI.getInfo(new ApiCallback<UserInfo>() {
                @Override
                public void onSuccess(UserInfo data) {
                    assertNotNull("用户信息不应为null", data);
                }

                @Override
                public void onError(int code, String message) {
                    // 业务错误是可以预期的
                }

                @Override
                public void onFailure(com.pei.dehaze.sdk.network.ApiException e) {
                    // 网络错误是可以预期的
                    assertNotNull("应该捕获到异常", e);
                }
            });

            // 测试获取用户分页列表
            UserQuery userQuery = new UserQuery();
            userQuery.setPageNum(1);
            userQuery.setPageSize(10);

            UserAPI.getPage(userQuery, new ApiCallback<PageResult<UserPageVO>>() {
                @Override
                public void onSuccess(PageResult<UserPageVO> data) {
                    assertNotNull("用户分页结果不应为null", data);
                }

                @Override
                public void onError(int code, String message) {
                    // 业务错误是可以预期的
                }

                @Override
                public void onFailure(com.pei.dehaze.sdk.network.ApiException e) {
                    // 网络错误是可以预期的
                    assertNotNull("应该捕获到异常", e);
                }
            });
        } catch (Exception e) {
            fail("用户API测试异常: " + e.getMessage());
        }
    }

    @Test
    public void testAlgorithmAPI() {
        // 测试算法相关API
        try {
            // 测试获取算法列表
            AlgorithmQuery algorithmQuery = new AlgorithmQuery();
            algorithmQuery.setKeywords("test");

            AlgorithmAPI.getList(algorithmQuery, new ApiCallback<List<Algorithm>>() {
                @Override
                public void onSuccess(List<Algorithm> data) {
                    assertNotNull("算法列表不应为null", data);
                }

                @Override
                public void onError(int code, String message) {
                    // 业务错误是可以预期的
                }

                @Override
                public void onFailure(com.pei.dehaze.sdk.network.ApiException e) {
                    // 网络错误是可以预期的
                    assertNotNull("应该捕获到异常", e);
                }
            });
        } catch (Exception e) {
            fail("算法API测试异常: " + e.getMessage());
        }
    }

    @Test
    public void testDatasetAPI() {
        // 测试数据集相关API
        try {
            // 测试获取数据集列表
            DatasetQuery datasetQuery = new DatasetQuery();
            datasetQuery.setKeywords("test");

            DatasetAPI.getList(datasetQuery, new ApiCallback<List<Dataset>>() {
                @Override
                public void onSuccess(List<Dataset> data) {
                    assertNotNull("数据集列表不应为null", data);
                }

                @Override
                public void onError(int code, String message) {
                    // 业务错误是可以预期的
                }

                @Override
                public void onFailure(com.pei.dehaze.sdk.network.ApiException e) {
                    // 网络错误是可以预期的
                    assertNotNull("应该捕获到异常", e);
                }
            });
        } catch (Exception e) {
            fail("数据集API测试异常: " + e.getMessage());
        }
    }

    @Test
    public void testDeptAPI() {
        // 测试部门相关API
        try {
            // 测试获取部门列表
            DeptQuery deptQuery = new DeptQuery();
            deptQuery.setKeywords("test");

            DeptAPI.getList(deptQuery, new ApiCallback<List<DeptVO>>() {
                @Override
                public void onSuccess(List<DeptVO> data) {
                    assertNotNull("部门列表不应为null", data);
                }

                @Override
                public void onError(int code, String message) {
                    // 业务错误是可以预期的
                }

                @Override
                public void onFailure(com.pei.dehaze.sdk.network.ApiException e) {
                    // 网络错误是可以预期的
                    assertNotNull("应该捕获到异常", e);
                }
            });
        } catch (Exception e) {
            fail("部门API测试异常: " + e.getMessage());
        }
    }

    @Test
    public void testDictAPI() {
        // 测试字典相关API
        try {
            // 测试获取字典类型分页列表
            DictTypeQuery dictTypeQuery = new DictTypeQuery();
            dictTypeQuery.setPageNum(1);
            dictTypeQuery.setPageSize(10);
            dictTypeQuery.setKeywords("test");

            DictAPI.getDictTypePage(dictTypeQuery, new ApiCallback<PageResult<DictTypePageVO>>() {
                @Override
                public void onSuccess(PageResult<DictTypePageVO> data) {
                    assertNotNull("字典类型分页结果不应为null", data);
                }

                @Override
                public void onError(int code, String message) {
                    // 业务错误是可以预期的
                }

                @Override
                public void onFailure(com.pei.dehaze.sdk.network.ApiException e) {
                    // 网络错误是可以预期的
                    assertNotNull("应该捕获到异常", e);
                }
            });
        } catch (Exception e) {
            fail("字典API测试异常: " + e.getMessage());
        }
    }

    @Test
    public void testMenuAPI() {
        // 测试菜单相关API
        try {
            // 测试获取菜单列表
            MenuQuery menuQuery = new MenuQuery();
            menuQuery.setKeywords("test");

            MenuAPI.getList(menuQuery, new ApiCallback<List<MenuVO>>() {
                @Override
                public void onSuccess(List<MenuVO> data) {
                    assertNotNull("菜单列表不应为null", data);
                }

                @Override
                public void onError(int code, String message) {
                    // 业务错误是可以预期的
                }

                @Override
                public void onFailure(com.pei.dehaze.sdk.network.ApiException e) {
                    // 网络错误是可以预期的
                    assertNotNull("应该捕获到异常", e);
                }
            });
        } catch (Exception e) {
            fail("菜单API测试异常: " + e.getMessage());
        }
    }

    @Test
    public void testRoleAPI() {
        // 测试角色相关API
        try {
            // 测试获取角色分页列表
            RoleQuery roleQuery = new RoleQuery();
            roleQuery.setPageNum(1);
            roleQuery.setPageSize(10);
            roleQuery.setKeywords("test");

            RoleAPI.getPage(roleQuery, new ApiCallback<PageResult<RolePageVO>>() {
                @Override
                public void onSuccess(PageResult<RolePageVO> data) {
                    assertNotNull("角色分页结果不应为null", data);
                }

                @Override
                public void onError(int code, String message) {
                    // 业务错误是可以预期的
                }

                @Override
                public void onFailure(com.pei.dehaze.sdk.network.ApiException e) {
                    // 网络错误是可以预期的
                    assertNotNull("应该捕获到异常", e);
                }
            });
        } catch (Exception e) {
            fail("角色API测试异常: " + e.getMessage());
        }
    }
}
