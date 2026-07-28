package com.pei.dehaze.config.property;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;

@Data
@ConfigurationProperties(prefix = "payment")
public class PaymentProperties {

    private ChannelConfig wechat = new ChannelConfig();
    private ChannelConfig alipay = new ChannelConfig();

    @Data
    public static class ChannelConfig {
        private boolean enabled = false;
        private String appId;
        private String mchId;
        private String apiKey;
        private String privateKey;
        private String publicKey;
        private String notifyUrl;
        private String refundNotifyUrl;
        private String baseUrl;
    }
}
