package com.pei.dehaze.service.impl.file;

import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.service.FileService;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.autoconfigure.context.ConfigurationPropertiesAutoConfiguration;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.FilterType;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * 存储后端激活与默认后端守卫。
 * <p>
 * 本模块的设计是**多后端共存**：{@link StorageServiceFactory} 收集全部 {@link FileService} bean 按
 * {@link FileService#getStorageType()} 索引（历史文件可能落在非默认后端），
 * {@code RatingServiceImpl}/{@code FeedbackServiceImpl} 也会遍历全部实现收集允许的图片 URL 前缀。
 * 因此**不得**把实现改为按 {@code file.type} 条件激活（那会让非默认后端消失，
 * 读取/校验随之失败）；{@code file.type} 的唯一语义是"新上传文件的默认后端"。
 * 本测试锁定该口径：三个后端始终共存，{@code getDefault()} 跟随 {@code file.type}。
 * <p>
 * 背景：曾有实现按各自子配置条件激活（`file.local.upload-path`/`file.minio.endpoint`/
 * `file.nginx-static.base-url` 在 dev 下同时成立）导致三个 bean 同时存在，
 * 而单实现注入点无法解析；修法是让单实现注入点改用 {@code StorageServiceFactory.getDefault()}，
 * 而不是收敛后端数量。
 *
 * @author earthyzinc
 * @since 2026-09-18
 */
@DisplayName("FileService 多后端共存与默认后端守卫")
class FileStorageBackendActivationTest {

    private final ApplicationContextRunner runner = new ApplicationContextRunner()
            .withConfiguration(AutoConfigurations.of(ConfigurationPropertiesAutoConfiguration.class))
            .withUserConfiguration(BackendScanConfig.class)
            .withPropertyValues(
                    "file.baseUrl=http://127.0.0.1:8989/api/v1/files/download",
                    "file.local.upload-path=target/test-upload",
                    // 指向必然拒连的端口：MinioFileService.init() 自带重试且不阻塞启动，此处只需快速失败
                    "file.minio.endpoint=http://127.0.0.1:1",
                    "file.minio.access-key=test",
                    "file.minio.secret-key=test",
                    "file.minio.bucket-name=test",
                    "file.nginx-static.base-url=http://127.0.0.1:9000");

    @ParameterizedTest(name = "file.type={0}：三后端共存，默认后端跟随 file.type")
    @ValueSource(strings = {"local", "minio", "nginx-static"})
    void allBackendsCoexist_andDefaultFollowsFileType(String type) {
        runner.withPropertyValues("file.type=" + type).run(context -> {
            assertThat(context).hasNotFailed();
            assertThat(context.getBeansOfType(FileService.class))
                    .as("三个存储后端必须同时可解析（多后端分发/URL 前缀校验依赖全量注册）")
                    .hasSize(3);
            assertThat(context.getBean(StorageServiceFactory.class).getDefault().getStorageType())
                    .as("默认上传后端由 file.type 决定")
                    .isEqualTo(type);
        });
    }

    @Test
    @DisplayName("file.type 取值非法时快速失败，不静默落到某个后端")
    void unknownFileType_failsFast() {
        runner.withPropertyValues("file.type=bogus").run(context -> {
            assertThat(context).hasNotFailed();
            assertThatThrownBy(() -> context.getBean(StorageServiceFactory.class).getDefault())
                    .isInstanceOf(BusinessException.class)
                    .hasMessageContaining("不支持的存储后端");
        });
    }

    /** 只扫描存储后端相关的实现（保持生产条件注解为唯一事实源，不在测试里重复条件） */
    @Configuration
    @ComponentScan(basePackageClasses = FileService.class, useDefaultFilters = false,
            includeFilters = @ComponentScan.Filter(type = FilterType.ASSIGNABLE_TYPE,
                    classes = {LocalFileService.class, MinioFileService.class, NginxStaticFileService.class,
                            StorageServiceFactory.class}))
    static class BackendScanConfig {
    }
}
