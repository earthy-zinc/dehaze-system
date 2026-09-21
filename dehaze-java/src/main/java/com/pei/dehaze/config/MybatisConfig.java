package com.pei.dehaze.config;

import com.baomidou.mybatisplus.annotation.DbType;
import com.baomidou.mybatisplus.extension.plugins.MybatisPlusInterceptor;
import com.baomidou.mybatisplus.extension.plugins.inner.DataPermissionInterceptor;
import com.baomidou.mybatisplus.extension.plugins.inner.PaginationInnerInterceptor;
import com.pei.dehaze.plugin.mybatis.handler.MyDataPermissionHandler;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.transaction.annotation.EnableTransactionManagement;

/**
 * mybatis-plus 配置类
 *
 * @author earthyzinc
 * @since 2022/7/2
 */
@Configuration
@EnableTransactionManagement
public class MybatisConfig {

    /** 单页条数兜底上限：正常越界由 {@code @Max(100)}+{@code @Valid} 报 A0400，此值兜住校验漏网的请求 */
    static final long PAGE_SIZE_MAX_LIMIT = 200L;

    /**
     * 分页插件和数据权限插件
     */
    @Bean
    public MybatisPlusInterceptor mybatisPlusInterceptor() {
        MybatisPlusInterceptor interceptor = new MybatisPlusInterceptor();
        //数据权限
        interceptor.addInnerInterceptor(new DataPermissionInterceptor(new MyDataPermissionHandler()));
        //分页插件：校验层与兜底层分层——参数校验缺失时（如新端点忘记 @Valid）在此静默截断，
        //避免 pageSize 越界直接打出大结果集拖垮 DB；负数 size 也会被 clamp（原先等价于不分页取全表）
        PaginationInnerInterceptor pagination = new PaginationInnerInterceptor(DbType.MYSQL);
        pagination.setMaxLimit(PAGE_SIZE_MAX_LIMIT);
        interceptor.addInnerInterceptor(pagination);

        return interceptor;
    }

}
