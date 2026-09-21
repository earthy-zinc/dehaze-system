package com.pei.dehaze.controller;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.security.access.prepost.PreAuthorize;

import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Agent 管理端点权限口径守卫（纯反射，不启 Spring 上下文）。
 *
 * <p>对齐 dehaze-python {@code router/ai_agent.py}：
 * <ul>
 *   <li>写操作与版本管理读接口（{@code _ensure_manager}，行 36-45，调用点 233/255/273/298）要求
 *       ROOT 或 {@code ai:agent:manage}——版本快照含完整提示词/权限/配置，仅管理端可见；</li>
 *   <li>列表/可选列表/详情/默认值仅需登录（无权限装饰器）。</li>
 * </ul>
 * 该口径曾被误判为"python 仅登录"而计划放宽，故用测试钉死，避免权限边界回退。
 * 发布（POST /{agentId}/publish，含回归评测门禁）归 B 类转发 python，其「原生控制器不得声明该路径」
 * 的守卫由 {@code AiProxyMappingTest} 统一承担（同一约束不在两处重复维护）。
 */
@DisplayName("Agent 端点权限口径守卫")
class AiAgentControllerPermissionTest {

    private static final String MANAGE_PERMISSION = "@ss.hasPerm('ai:agent:manage')";

    private static Method method(String name) {
        return Arrays.stream(AiAgentController.class.getMethods())
                .filter(candidate -> candidate.getName().equals(name))
                .findFirst()
                .orElseThrow(() -> new AssertionError("方法不存在: " + name));
    }

    private static void assertRequiresManage(String name) {
        PreAuthorize annotation = method(name).getAnnotation(PreAuthorize.class);
        assertThat(annotation).as("%s 应要求 ai:agent:manage（对齐 python _ensure_manager）", name).isNotNull();
        assertThat(annotation.value()).as(name).isEqualTo(MANAGE_PERMISSION);
    }

    @Test
    @DisplayName("写操作全部要求 ai:agent:manage（发布已移交转发域，不在原生控制器）")
    void writeEndpointsRequireManagePermission() {
        List.of("create", "update", "delete", "setStatus", "copy", "setSkills", "setMcps",
                "setSubagents", "rollback").forEach(AiAgentControllerPermissionTest::assertRequiresManage);
    }

    @Test
    @DisplayName("版本历史/差异/快照详情要求 ai:agent:manage（版本快照含完整提示词，仅管理端可见）")
    void versionEndpointsRequireManagePermission() {
        List.of("versions", "diffVersions", "versionDetail")
                .forEach(AiAgentControllerPermissionTest::assertRequiresManage);
    }

    @Test
    @DisplayName("列表/可选列表/详情/推理参数默认值仅需登录")
    void readEndpointsRequireLoginOnly() {
        for (String name : List.of("list", "listEnabled", "configDefaults", "detail")) {
            assertThat(method(name).getAnnotation(PreAuthorize.class)).as(name).isNull();
        }
    }
}
