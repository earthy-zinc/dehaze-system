package com.pei.dehaze.framework.idempotent.config;

import com.pei.dehaze.framework.idempotent.core.aop.IdempotentAspect;
import com.pei.dehaze.framework.idempotent.core.keyresolver.impl.DefaultIdempotentKeyResolver;
import com.pei.dehaze.framework.idempotent.core.keyresolver.impl.ExpressionIdempotentKeyResolver;
import com.pei.dehaze.framework.idempotent.core.keyresolver.IdempotentKeyResolver;
import com.pei.dehaze.framework.idempotent.core.keyresolver.impl.UserIdempotentKeyResolver;
import com.pei.dehaze.framework.idempotent.core.redis.IdempotentRedisDAO;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import com.pei.dehaze.framework.redis.config.PeiRedisAutoConfiguration;
import org.springframework.context.annotation.Bean;
import org.springframework.data.redis.core.StringRedisTemplate;

import java.util.List;

@AutoConfiguration(after = PeiRedisAutoConfiguration.class)
public class PeiIdempotentConfiguration {

    @Bean
    public IdempotentAspect idempotentAspect(List<IdempotentKeyResolver> keyResolvers, IdempotentRedisDAO idempotentRedisDAO) {
        return new IdempotentAspect(keyResolvers, idempotentRedisDAO);
    }

    @Bean
    public IdempotentRedisDAO idempotentRedisDAO(StringRedisTemplate stringRedisTemplate) {
        return new IdempotentRedisDAO(stringRedisTemplate);
    }

    // ========== 各种 IdempotentKeyResolver Bean ==========

    @Bean
    public DefaultIdempotentKeyResolver defaultIdempotentKeyResolver() {
        return new DefaultIdempotentKeyResolver();
    }

    @Bean
    public UserIdempotentKeyResolver userIdempotentKeyResolver() {
        return new UserIdempotentKeyResolver();
    }

    @Bean
    public ExpressionIdempotentKeyResolver expressionIdempotentKeyResolver() {
        return new ExpressionIdempotentKeyResolver();
    }

}
