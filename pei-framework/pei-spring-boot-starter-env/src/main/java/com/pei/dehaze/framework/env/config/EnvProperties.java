package com.pei.dehaze.framework.env.config;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * 环境配置
 *
 * @author earthyzinc
 */
@ConfigurationProperties(prefix = "pei.env")
@Data
public class EnvProperties {

    public static final String TAG_KEY = "pei.env.tag";

    /**
     * 环境标签
     */
    private String tag;

}
