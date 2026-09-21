package com.pei.dehaze.controller;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.config.BeanDefinition;
import org.springframework.context.annotation.ClassPathScanningCandidateComponentProvider;
import org.springframework.core.annotation.AnnotatedElementUtils;
import org.springframework.stereotype.Controller;
import org.springframework.util.AntPathMatcher;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Controller 路由唯一性守卫。
 * <p>
 * Spring 对「相同路径 + 相同方法」的两处映射会在启动时抛 Ambiguous mapping，
 * 而多成员并行开发时最容易出现两个 Controller 声明同一端点（如全局 Agent Card）。
 * 本测试在单测阶段就暴露这类冲突，避免拖到应用启动才失败。
 */
@DisplayName("Controller 路由唯一性守卫")
class ControllerMappingUniquenessTest {

    private static final String BASE_PACKAGE = "com.pei.dehaze.controller";

    @Test
    @DisplayName("同一「方法 + 路径」不得被两个 Controller 重复声明")
    void noDuplicateMappings() throws Exception {
        Map<String, List<String>> owners = new LinkedHashMap<>();
        List<String> duplicates = new ArrayList<>();

        for (Class<?> controller : controllers()) {
            for (Map.Entry<String, String> mapping : mappingsOf(controller).entrySet()) {
                List<String> declaredBy = owners.computeIfAbsent(mapping.getKey(), key -> new ArrayList<>());
                declaredBy.add(mapping.getValue());
                if (declaredBy.size() == 2) {
                    duplicates.add(mapping.getKey() + " 由 " + declaredBy + " 重复声明");
                }
            }
        }

        assertThat(duplicates)
                .as("重复路由会导致 Spring 启动失败（Ambiguous mapping）")
                .isEmpty();
    }

    private static List<Class<?>> controllers() throws ClassNotFoundException {
        ClassPathScanningCandidateComponentProvider scanner =
                new ClassPathScanningCandidateComponentProvider(false);
        scanner.addIncludeFilter(new org.springframework.core.type.filter.AnnotationTypeFilter(RestController.class));
        scanner.addIncludeFilter(new org.springframework.core.type.filter.AnnotationTypeFilter(Controller.class));
        List<Class<?>> controllers = new ArrayList<>();
        for (BeanDefinition definition : scanner.findCandidateComponents(BASE_PACKAGE)) {
            Class<?> candidate = Class.forName(definition.getBeanClassName());
            // 排除测试内嵌的模拟 Controller（真实 Controller 均为顶层类）
            if (!candidate.isMemberClass()) {
                controllers.add(candidate);
            }
        }
        return controllers;
    }

    /** 展开 Controller 的全部「HTTP 方法 + 路径」组合 */
    private static Map<String, String> mappingsOf(Class<?> controller) {
        AntPathMatcher matcher = new AntPathMatcher();
        List<String> classPaths = pathsOf(AnnotatedElementUtils.findMergedAnnotation(controller, RequestMapping.class));
        Map<String, String> mappings = new LinkedHashMap<>();

        for (Method method : controller.getMethods()) {
            RequestMapping methodMapping = AnnotatedElementUtils.findMergedAnnotation(method, RequestMapping.class);
            if (methodMapping == null) {
                continue;
            }
            List<String> methodPaths = pathsOf(methodMapping);
            List<String> combined = new ArrayList<>();
            if (classPaths.isEmpty()) {
                combined.addAll(methodPaths);
            } else {
                for (String classPath : classPaths) {
                    for (String methodPath : methodPaths) {
                        combined.add(matcher.combine(classPath, methodPath));
                    }
                }
            }
            for (String path : combined) {
                for (String httpMethod : httpMethodsOf(methodMapping)) {
                    mappings.put(httpMethod + " " + path, controller.getSimpleName() + "#" + method.getName());
                }
            }
        }
        return mappings;
    }

    private static List<String> pathsOf(RequestMapping mapping) {
        if (mapping == null) {
            return List.of();
        }
        String[] paths = mapping.path().length > 0 ? mapping.path() : mapping.value();
        return List.of(paths).isEmpty() ? List.of("") : List.of(paths);
    }

    private static List<String> httpMethodsOf(RequestMapping mapping) {
        RequestMethod[] methods = mapping.method();
        if (methods.length == 0) {
            return List.of("ALL");
        }
        return List.of(methods).stream().map(Enum::name).toList();
    }
}
