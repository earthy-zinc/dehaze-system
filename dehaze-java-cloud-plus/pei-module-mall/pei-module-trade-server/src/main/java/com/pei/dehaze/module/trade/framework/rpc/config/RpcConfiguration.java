package com.pei.dehaze.module.trade.framework.rpc.config;

import com.pei.dehaze.module.member.api.address.MemberAddressApi;
import com.pei.dehaze.module.member.api.config.MemberConfigApi;
import com.pei.dehaze.module.member.api.level.MemberLevelApi;
import com.pei.dehaze.module.member.api.point.MemberPointApi;
import com.pei.dehaze.module.member.api.user.MemberUserApi;
import com.pei.dehaze.module.pay.api.order.PayOrderApi;
import com.pei.dehaze.module.pay.api.refund.PayRefundApi;
import com.pei.dehaze.module.pay.api.transfer.PayTransferApi;
import com.pei.dehaze.module.pay.api.wallet.PayWalletApi;
import com.pei.dehaze.module.product.api.category.ProductCategoryApi;
import com.pei.dehaze.module.product.api.comment.ProductCommentApi;
import com.pei.dehaze.module.product.api.sku.ProductSkuApi;
import com.pei.dehaze.module.product.api.spu.ProductSpuApi;
import com.pei.dehaze.module.promotion.api.bargain.BargainActivityApi;
import com.pei.dehaze.module.promotion.api.bargain.BargainRecordApi;
import com.pei.dehaze.module.promotion.api.combination.CombinationRecordApi;
import com.pei.dehaze.module.promotion.api.coupon.CouponApi;
import com.pei.dehaze.module.promotion.api.discount.DiscountActivityApi;
import com.pei.dehaze.module.promotion.api.point.PointActivityApi;
import com.pei.dehaze.module.promotion.api.reward.RewardActivityApi;
import com.pei.dehaze.module.promotion.api.seckill.SeckillActivityApi;
import com.pei.dehaze.module.system.api.notify.NotifyMessageSendApi;
import com.pei.dehaze.module.system.api.social.SocialClientApi;
import com.pei.dehaze.module.system.api.social.SocialUserApi;
import com.pei.dehaze.module.system.api.user.AdminUserApi;
import org.springframework.cloud.openfeign.EnableFeignClients;
import org.springframework.context.annotation.Configuration;

@Configuration(value = "tradeRpcConfiguration", proxyBeanMethods = false)
@EnableFeignClients(clients = {
        BargainActivityApi.class, BargainRecordApi.class, CombinationRecordApi.class,
        CouponApi.class, DiscountActivityApi.class, RewardActivityApi.class, SeckillActivityApi.class, PointActivityApi.class,
        MemberUserApi.class, MemberPointApi.class, MemberLevelApi.class, MemberAddressApi.class, MemberConfigApi.class,
        ProductSpuApi.class, ProductSkuApi.class, ProductCommentApi.class, ProductCategoryApi.class,
        PayOrderApi.class, PayRefundApi.class, PayTransferApi.class, PayWalletApi.class,
        AdminUserApi.class, NotifyMessageSendApi.class, SocialClientApi.class, SocialUserApi.class
})
public class RpcConfiguration {
}
