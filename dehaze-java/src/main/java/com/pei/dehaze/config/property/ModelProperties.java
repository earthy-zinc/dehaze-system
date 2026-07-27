package com.pei.dehaze.config.property;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

/**
 * 模型权重文件存储配置
 *
 * 复用 nginx-dataset 静态服务（/models 路径）托管算法权重文件，
 * Java 后端通过 HTTP HEAD 校验 sys_algorithm.path 字段可访问性并回填 size。
 *
 * @author earthy-zinc
 */
@Data
@Component
@ConfigurationProperties(prefix = "algorithm.model")
public class ModelProperties {

    /** Nginx 模型服务基础 URL，权重访问 URL = {baseUrl}/{algorithm.path} */
    private String baseUrl = "http://127.0.0.1:9000/models";

    /** HTTP HEAD 校验连接超时（毫秒） */
    private int connectTimeout = 5000;

    /** HTTP HEAD 校验读取超时（毫秒） */
    private int readTimeout = 10000;
}
