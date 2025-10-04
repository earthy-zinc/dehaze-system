package com.pei.dehaze.module.promotion.framework.rpc.config;

import com.pei.dehaze.module.infra.api.websocket.WebSocketSenderApi;
import com.pei.dehaze.module.member.api.user.MemberUserApi;
import com.pei.dehaze.module.product.api.category.ProductCategoryApi;
import com.pei.dehaze.module.product.api.sku.ProductSkuApi;
import com.pei.dehaze.module.product.api.spu.ProductSpuApi;
import com.pei.dehaze.module.system.api.social.SocialClientApi;
import com.pei.dehaze.module.system.api.user.AdminUserApi;
import com.pei.dehaze.module.trade.api.order.TradeOrderApi;
import org.springframework.cloud.openfeign.EnableFeignClients;
import org.springframework.context.annotation.Configuration;

@Configuration(value = "promotionRpcConfiguration", proxyBeanMethods = false)
@EnableFeignClients(clients = {ProductSkuApi.class, ProductSpuApi.class, ProductCategoryApi.class,
        MemberUserApi.class, TradeOrderApi.class, AdminUserApi.class, SocialClientApi.class,
        WebSocketSenderApi.class})
public class RpcConfiguration {
}
