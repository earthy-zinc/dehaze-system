package com.pei.dehaze.controller;

import com.pei.dehaze.service.AiA2aService;
import com.pei.dehaze.service.client.AiForwardClient;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.annotation.AnnotatedElementUtils;
import org.springframework.mock.web.MockHttpServletRequest;
import org.springframework.mock.web.MockServletContext;
import org.springframework.util.AntPathMatcher;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.context.support.AnnotationConfigWebApplicationContext;
import org.springframework.web.servlet.HandlerExecutionChain;
import org.springframework.web.servlet.config.annotation.EnableWebMvc;
import org.springframework.web.servlet.mvc.method.annotation.RequestMappingHandlerMapping;

import java.lang.reflect.Method;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;

/**
 * 转发通配映射与 A 类/原生精确映射的共存性测试。
 * <p>
 * 转发入口以 {@code /api/v1/**} 通配注册，若与任何精确映射的「路径 + 方法」组合撞车，
 * Spring 启动时会因歧义映射直接失败；本测试同时锁定"精确映射优先于通配"的路由归属。
 * <p>
 * 归属守卫（原生 Controller 不得声明必须转发 python 的路径）有两处：publish（{@link AiAgentController}，
 * 由 java-crud-1 的权限测试让位而来）与知识库转发专属端点（{@link AiKnowledgeBaseController} 的
 * {@code PUT|DELETE /api/v1/kb/documents/{id}} 及库级 {@code POST /api/v1/kb}、
 * {@code DELETE /api/v1/kb/{id}}、{@code GET /api/v1/kb/{id}/index-stats}）；重构或移除这些断言前
 * 须与原作者确认，否则对应路径可能被原生实现截胡（转发静默失效）且无人守卫。
 */
@DisplayName("AiProxyController 映射冲突防护测试")
class AiProxyMappingTest {

    @Test
    @DisplayName("上下文可加载（无歧义映射），且精确映射优先命中")
    void wildcardProxyCoexistsWithExactMappings() throws Exception {
        try (AnnotationConfigWebApplicationContext context = new AnnotationConfigWebApplicationContext()) {
            context.setServletContext(new MockServletContext());
            context.register(WebConfig.class);
            context.refresh();

            RequestMappingHandlerMapping mapping = context.getBean(RequestMappingHandlerMapping.class);

            // A 类 CRUD：消息通知模块的精确映射应由业务 Controller 承接，不被转发入口截胡
            assertThat(handlerType(mapping, "GET", "/api/v1/messages")).isEqualTo(NotificationController.class);
            assertThat(handlerType(mapping, "POST", "/api/v1/messages/send")).isEqualTo(NotificationController.class);
            // B 类端点：通配转发入口承接
            assertThat(handlerType(mapping, "GET", "/api/v1/ai/mcp/servers/7/health")).isEqualTo(AiProxyController.class);
            assertThat(handlerType(mapping, "POST", "/api/v1/ai/agents/5/a2a")).isEqualTo(AiProxyController.class);
            // 发布门禁依赖 LLM，java 侧不保留原生实现，由转发入口承接
            assertThat(handlerType(mapping, "POST", "/api/v1/ai/agents/5/publish")).isEqualTo(AiProxyController.class);
            assertThat(handlerType(mapping, "POST", "/a2a")).isEqualTo(AiProxyController.class);
            assertThat(handlerType(mapping, "GET", "/.well-known/agent.json")).isEqualTo(AiProxyController.class);
            // 文档版本更新/删除：原生实现不驱动 python 的 _process_document_guarded，已移除，由通配转发承接
            assertThat(handlerType(mapping, "PUT", "/api/v1/kb/documents/7")).isEqualTo(AiProxyController.class);
            assertThat(handlerType(mapping, "DELETE", "/api/v1/kb/documents/7")).isEqualTo(AiProxyController.class);
            // 库级创建/删除/索引统计（ES 索引生命周期）同样由转发入口承接，且 /api/v1/** 必须能命中无尾段的 /api/v1/kb
            assertThat(handlerType(mapping, "POST", "/api/v1/kb")).isEqualTo(AiProxyController.class);
            assertThat(handlerType(mapping, "DELETE", "/api/v1/kb/7")).isEqualTo(AiProxyController.class);
            assertThat(handlerType(mapping, "GET", "/api/v1/kb/7/index-stats")).isEqualTo(AiProxyController.class);
            // 语音域：ASR/TTS 引擎只在 python 进程内，java 无原生实现，四类形态（JSON/multipart/
            // 二进制响应/无尾段静态兄弟）都须由通配转发承接
            assertThat(handlerType(mapping, "POST", "/api/v1/voice/tts")).isEqualTo(AiProxyController.class);
            assertThat(handlerType(mapping, "GET", "/api/v1/voice/tts/voices")).isEqualTo(AiProxyController.class);
            assertThat(handlerType(mapping, "GET", "/api/v1/voice/tts/audio/abc123")).isEqualTo(AiProxyController.class);
            assertThat(handlerType(mapping, "POST", "/api/v1/voice/asr/offline")).isEqualTo(AiProxyController.class);
            assertThat(handlerType(mapping, "GET", "/api/v1/voice/hotwords/global")).isEqualTo(AiProxyController.class);
            assertThat(handlerType(mapping, "GET", "/api/v1/voice/providers/enabled")).isEqualTo(AiProxyController.class);
            assertThat(handlerType(mapping, "PUT", "/api/v1/voice/providers/3/keys/5")).isEqualTo(AiProxyController.class);
            // 挂载路径 Agent Card：java 原生实现落在 /api/v1/** 通配范围内，精确映射优先命中
            assertThat(handlerType(mapping, "GET", "/api/v1/ai/agents/5/a2a/.well-known/agent.json"))
                    .isEqualTo(AiA2aController.class);
        }
    }

    @Test
    @DisplayName("发布门禁归转发入口：AiAgentController 不得声明能命中 publish 的映射")
    void publishMapping_isNotDeclaredByNativeAgentController() {
        // publish 的发布门禁会真实执行回归评测（LLM 依赖），java 原生实现只能在配置含考题回归集时失败，
        // 故该路径必须由 AiProxyController 转发 python；此断言防止原生实现被回加后又被精确映射优先命中
        // 检测器自证：对已知存在的原生映射必须命中，否则"断言不存在"会假绿
        assertThat(declaresMappingFor(NotificationController.class, "POST", "/api/v1/messages/send"))
                .as("检测器自证")
                .isTrue();
        assertThat(declaresMappingFor(AiAgentController.class, "POST", "/api/v1/ai/agents/5/publish"))
                .as("AiAgentController 不得声明 publish 原生映射（精确映射会优先于 /api/v1/** 转发通配）")
                .isFalse();
    }

    @Test
    @DisplayName("知识库转发专属端点归转发入口：AiKnowledgeBaseController 不得声明这些映射")
    void kbForwardOnlyMappings_areNotDeclaredByNativeKbController() {
        // 这些端点必须由转发入口承接（依赖 python 的文档流水线或 ES 索引生命周期）；
        // 一旦原生被回加，精确映射会优先于 /api/v1/** 转发通配，转发即失效——此断言负责拦截
        // 检测器自证：同一控制器的库详情与文档详情 GET 映射必须命中
        assertThat(declaresMappingFor(AiKnowledgeBaseController.class, "GET", "/api/v1/kb/documents/7"))
                .as("检测器自证（文档详情）")
                .isTrue();
        assertThat(declaresMappingFor(AiKnowledgeBaseController.class, "GET", "/api/v1/kb/7"))
                .as("检测器自证（库详情）")
                .isTrue();
        // 文档版本更新/删除：须驱动 python 的 _process_document_guarded，原生只改状态会让 processingStatus 永久 pending
        assertThat(declaresMappingFor(AiKnowledgeBaseController.class, "PUT", "/api/v1/kb/documents/7"))
                .as("文档版本更新必须转发 python（原生实现会让 processingStatus 永久 pending）")
                .isFalse();
        assertThat(declaresMappingFor(AiKnowledgeBaseController.class, "DELETE", "/api/v1/kb/documents/7"))
                .as("文档删除必须转发 python（原生实现不驱动向量清理与状态机）")
                .isFalse();
        // 库级创建/删除/索引统计：依赖 ES 索引生命周期（ensure_kb_index / delete_kb_index / _stats）
        assertThat(declaresMappingFor(AiKnowledgeBaseController.class, "POST", "/api/v1/kb"))
                .as("库创建必须转发 python（原生实现会出现「库建了但 ES 索引不存在」）")
                .isFalse();
        assertThat(declaresMappingFor(AiKnowledgeBaseController.class, "DELETE", "/api/v1/kb/7"))
                .as("库删除必须转发 python（须同步删除 ES 索引）")
                .isFalse();
        assertThat(declaresMappingFor(AiKnowledgeBaseController.class, "GET", "/api/v1/kb/7/index-stats"))
                .as("索引状态必须转发 python（读 ES _stats）")
                .isFalse();
    }

    /** 该 Controller 是否声明了能命中「HTTP 方法 + 具体请求路径」的映射（含变量名差异与通配） */
    private static boolean declaresMappingFor(Class<?> controller, String method, String concretePath) {
        AntPathMatcher matcher = new AntPathMatcher();
        List<String> classPaths = patternsOf(AnnotatedElementUtils.findMergedAnnotation(controller, RequestMapping.class));
        for (Method candidate : controller.getMethods()) {
            RequestMapping methodMapping = AnnotatedElementUtils.findMergedAnnotation(candidate, RequestMapping.class);
            if (methodMapping == null || !acceptsMethod(methodMapping, method)) {
                continue;
            }
            for (String classPath : classPaths) {
                for (String methodPath : patternsOf(methodMapping)) {
                    if (matcher.match(matcher.combine(classPath, methodPath), concretePath)) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    private static List<String> patternsOf(RequestMapping mapping) {
        if (mapping == null) {
            return List.of("");
        }
        String[] patterns = mapping.path().length > 0 ? mapping.path() : mapping.value();
        return patterns.length > 0 ? List.of(patterns) : List.of("");
    }

    private static boolean acceptsMethod(RequestMapping mapping, String method) {
        return mapping.method().length == 0
                || List.of(mapping.method()).stream().anyMatch(candidate -> candidate.name().equals(method));
    }

    private static Class<?> handlerType(RequestMappingHandlerMapping mapping, String method, String uri)
            throws Exception {
        MockHttpServletRequest request = new MockHttpServletRequest(method, uri);
        HandlerExecutionChain chain = mapping.getHandler(request);
        assertThat(chain).as("未匹配到处理器: %s %s", method, uri).isNotNull();
        return ((org.springframework.web.method.HandlerMethod) chain.getHandler()).getBeanType();
    }

    @Configuration
    @EnableWebMvc
    static class WebConfig {

        @Bean
        AiProxyController aiProxyController() {
            return new AiProxyController(mock(AiForwardClient.class));
        }

        @Bean
        AiA2aController aiA2aController() {
            return new AiA2aController(mock(AiA2aService.class));
        }

        @Bean
        NotificationController notificationController() {
            return new NotificationController();
        }
    }

    /** 模拟其他模块（如消息通知）已有的精确路径映射 */
    @RestController
    @RequestMapping("/api/v1/messages")
    static class NotificationController {

        @GetMapping
        public String list() {
            return "list";
        }

        @PostMapping("/send")
        public String send() {
            return "send";
        }
    }
}
