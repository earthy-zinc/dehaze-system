package com.pei.dehaze.module.bpm.framework.web.config;

import com.pei.dehaze.framework.common.enums.WebFilterOrderEnum;
import com.pei.dehaze.module.bpm.framework.web.core.FlowableWebFilter;
import org.springframework.boot.web.servlet.FilterRegistrationBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * bpm 模块的 web 组件的 Configuration
 *
 * @author earthyzinc
 */
@Configuration(proxyBeanMethods = false)
public class BpmWebConfiguration {

    /**
     * 配置 Flowable Web 过滤器
     */
    @Bean
    public FilterRegistrationBean<FlowableWebFilter> flowableWebFilter() {
        FilterRegistrationBean<FlowableWebFilter> registrationBean = new FilterRegistrationBean<>();
        registrationBean.setFilter(new FlowableWebFilter());
        registrationBean.setOrder(WebFilterOrderEnum.FLOWABLE_FILTER);
        return registrationBean;
    }

}
