package com.pei.dehaze.service.impl.file;

import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.model.bo.FileBO;
import com.pei.dehaze.service.FileService;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;

/**
 * nginx-static 存储后端。
 * 文件实际由外部 nginx 静态服务托管，Java 端不参与上传，
 * 仅用于为 sys_file.storage = "nginx-static" 的记录在运行时拼接可访问 URL。
 * 适用于由算法/外部系统直写 nginx 静态目录、Java 仅做登记与 URL 拼接的场景。
 * 下载场景：FileController 直接 302 跳转；导出打包等需要文件流的场景通过 HTTP GET 拉取。
 *
 * @author earthyzinc
 */
@Component
@ConditionalOnProperty(prefix = "file.nginx-static", name = "base-url")
@ConfigurationProperties(prefix = "file.nginx-static")
@Data
@Slf4j
public class NginxStaticFileService implements FileService {

    private String baseUrl;

    private final HttpClient httpClient = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(10))
            .build();

    @Override
    public String getStorageType() {
        return "nginx-static";
    }

    @Override
    public String getBaseUrl() {
        return baseUrl;
    }

    @Override
    public FileBO uploadFile(FileBO fileBO) {
        throw new BusinessException("nginx-static 后端不支持通过 Java 上传文件");
    }

    @Override
    public String uploadFile(String objectName, InputStream inputStream, long fileSize, String contentType) {
        throw new BusinessException("nginx-static 后端不支持通过 Java 上传文件");
    }

    @Override
    public boolean deleteFile(String objectName) {
        throw new BusinessException("nginx-static 后端不支持通过 Java 删除文件");
    }

    @Override
    public InputStream downLoadFile(String objectName) {
        try {
            HttpResponse<InputStream> response = httpClient.send(
                    HttpRequest.newBuilder()
                            .uri(URI.create(getUrl(objectName)))
                            .timeout(Duration.ofSeconds(60))
                            .GET()
                            .build(),
                    HttpResponse.BodyHandlers.ofInputStream());
            if (response.statusCode() != 200) {
                throw new BusinessException("nginx-static 下载失败: " + response.statusCode() + ", objectName=" + objectName);
            }
            return response.body();
        } catch (IOException | InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new BusinessException("nginx-static 下载失败: " + e.getMessage(), e);
        }
    }
}
