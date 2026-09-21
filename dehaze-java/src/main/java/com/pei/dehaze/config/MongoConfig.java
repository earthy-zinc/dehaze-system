package com.pei.dehaze.config;

import com.pei.dehaze.model.entity.AuditLog;
import com.pei.dehaze.model.entity.LoginLog;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.ApplicationRunner;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.domain.Sort;
import org.springframework.data.mongodb.core.MongoTemplate;
import org.springframework.data.mongodb.core.index.Index;

@Slf4j
@Configuration
@RequiredArgsConstructor
public class MongoConfig {

    private final MongoTemplate mongoTemplate;

    @Bean
    public ApplicationRunner mongoIndexInitializer() {
        return args -> {
            // 索引键名与实体 @Field 映射后的 snake_case 一致（集合与 python 共用）
            mongoTemplate.indexOps(LoginLog.class).ensureIndex(
                    new Index().on("user_id", Sort.Direction.ASC).on("create_time", Sort.Direction.DESC));
            mongoTemplate.indexOps(LoginLog.class).ensureIndex(
                    new Index().on("create_time", Sort.Direction.DESC));
            mongoTemplate.indexOps(LoginLog.class).ensureIndex(
                    new Index().on("status", Sort.Direction.ASC));
            mongoTemplate.indexOps(AuditLog.class).ensureIndex(
                    new Index().on("operator_id", Sort.Direction.ASC).on("create_time", Sort.Direction.DESC));
            mongoTemplate.indexOps(AuditLog.class).ensureIndex(
                    new Index().on("target_type", Sort.Direction.ASC).on("target_id", Sort.Direction.ASC).on("create_time", Sort.Direction.DESC));
            mongoTemplate.indexOps(AuditLog.class).ensureIndex(
                    new Index().on("module", Sort.Direction.ASC).on("create_time", Sort.Direction.DESC));
            log.info("MongoDB 索引初始化完成");
        };
    }
}
