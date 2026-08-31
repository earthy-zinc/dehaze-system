package com.pei.dehaze.controller;

import com.pei.dehaze.service.impl.file.MinioFileService;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.bson.Document;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.mongodb.core.MongoTemplate;
import org.springframework.data.redis.connection.RedisConnection;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.http.ResponseEntity;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import javax.sql.DataSource;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * 健康检查端点
 *
 * 提供 K8s 探针使用的 liveness/readiness 分离端点：
 * - /health（liveness）：始终 200，仅表示进程存活
 * - /ready（readiness）：检查 DB/Redis/RabbitMQ 依赖，任一不可用返回 503
 */
@Tag(name = "健康检查")
@RestController
@RequiredArgsConstructor
@Slf4j
public class HealthController {

    private final DataSource dataSource;
    private final RedisTemplate<String, Object> redisTemplate;
    private final RabbitTemplate rabbitTemplate;
    private final MongoTemplate mongoTemplate;

    /**
     * MinIO 仅当配置了 file.minio.endpoint 时启用（MinioFileService 条件装配），
     * 未配置时该字段为 null，健康检查跳过 MinIO。
     */
    @Autowired(required = false)
    private MinioFileService minioFileService;

    /**
     * Liveness 探针 - 进程存活检查
     * 始终返回 200，仅表示进程正在运行
     */
    @GetMapping("/health")
    public ResponseEntity<Map<String, Object>> liveness() {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("status", "UP");
        return ResponseEntity.ok(body);
    }

    /**
     * Readiness 探针 - 就绪检查
     * 检查 DB/Redis/RabbitMQ 依赖，任一不可用返回 503
     */
    @GetMapping("/ready")
    public ResponseEntity<Map<String, Object>> readiness() {
        Map<String, Object> components = new LinkedHashMap<>();
        boolean allHealthy = true;

        // DB check
        try {
            new JdbcTemplate(dataSource).queryForObject("SELECT 1", Integer.class);
            components.put("db", "UP");
        } catch (Exception e) {
            components.put("db", "DOWN");
            allHealthy = false;
        }

        // Redis check
        try (RedisConnection conn = redisTemplate.getConnectionFactory().getConnection()) {
            conn.ping();
            components.put("redis", "UP");
        } catch (Exception e) {
            components.put("redis", "DOWN");
            allHealthy = false;
        }

        // RabbitMQ check
        try {
            rabbitTemplate.execute(channel -> null);
            components.put("rabbitmq", "UP");
        } catch (Exception e) {
            log.warn("RabbitMQ health check failed", e);
            components.put("rabbitmq", "DOWN");
            allHealthy = false;
        }

        // MongoDB check（必选基础设施）
        try {
            mongoTemplate.executeCommand(new Document("ping", 1));
            components.put("mongodb", "UP");
        } catch (Exception e) {
            log.warn("MongoDB health check failed", e);
            components.put("mongodb", "DOWN");
            allHealthy = false;
        }

        // MinIO check（仅当配置了 file.minio.endpoint 时）
        if (minioFileService != null) {
            if (minioFileService.ping()) {
                components.put("minio", "UP");
            } else {
                components.put("minio", "DOWN");
                allHealthy = false;
            }
        }

        Map<String, Object> body = new LinkedHashMap<>();
        body.put("status", allHealthy ? "UP" : "DOWN");
        body.put("components", components);

        if (!allHealthy) {
            return ResponseEntity.status(503).body(body);
        }
        return ResponseEntity.ok(body);
    }
}
