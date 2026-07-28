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
            mongoTemplate.indexOps(LoginLog.class).ensureIndex(
                    new Index().on("userId", Sort.Direction.ASC).on("createTime", Sort.Direction.DESC));
            mongoTemplate.indexOps(LoginLog.class).ensureIndex(
                    new Index().on("createTime", Sort.Direction.DESC));
            mongoTemplate.indexOps(LoginLog.class).ensureIndex(
                    new Index().on("status", Sort.Direction.ASC));
            mongoTemplate.indexOps(AuditLog.class).ensureIndex(
                    new Index().on("operatorId", Sort.Direction.ASC).on("createTime", Sort.Direction.DESC));
            mongoTemplate.indexOps(AuditLog.class).ensureIndex(
                    new Index().on("targetType", Sort.Direction.ASC).on("targetId", Sort.Direction.ASC).on("createTime", Sort.Direction.DESC));
            mongoTemplate.indexOps(AuditLog.class).ensureIndex(
                    new Index().on("module", Sort.Direction.ASC).on("createTime", Sort.Direction.DESC));
            log.info("MongoDB 索引初始化完成");
        };
    }
}
