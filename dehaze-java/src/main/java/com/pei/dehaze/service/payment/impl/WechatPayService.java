package com.pei.dehaze.service.payment.impl;

import com.pei.dehaze.config.property.PaymentProperties;
import com.pei.dehaze.service.payment.PaymentChannelService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.Map;
import java.util.TreeMap;

@Slf4j
@Service
@RequiredArgsConstructor
public class WechatPayService implements PaymentChannelService {

    private static final String CHANNEL_TYPE = "wechat";
    private static final String UNIFIED_ORDER_PATH = "/v3/pay/transactions/native";
    private static final String REFUND_PATH = "/v3/refund/domestic/refunds";
    private static final String CLOSE_ORDER_PATH = "/v3/pay/transactions/out-trade-no/%s/close";

    private final PaymentProperties paymentProperties;
    @Qualifier("paymentRestTemplate")
    private final RestTemplate restTemplate;

    @Override
    public String getChannelType() {
        return CHANNEL_TYPE;
    }

    @Override
    public boolean isEnabled() {
        return paymentProperties.getWechat().isEnabled();
    }

    @Override
    public UnifiedOrderResult unifiedOrder(String orderNo, long amountFen, String description, Map<String, String> extra) {
        if (!isEnabled()) {
            return mockOrder(orderNo);
        }
        try {
            PaymentProperties.ChannelConfig cfg = paymentProperties.getWechat();
            String url = cfg.getBaseUrl() + UNIFIED_ORDER_PATH;
            Map<String, Object> body = Map.of(
                    "appid", cfg.getAppId(),
                    "mchid", cfg.getMchId(),
                    "description", description,
                    "out_trade_no", orderNo,
                    "notify_url", cfg.getNotifyUrl(),
                    "amount", Map.of("total", amountFen, "currency", "CNY")
            );
            ResponseEntity<Map> resp = restTemplate.postForEntity(url, body, Map.class);
            if (resp.getStatusCode().is2xxSuccessful() && resp.getBody() != null) {
                String codeUrl = (String) resp.getBody().get("code_url");
                return new UnifiedOrderResult(true, null, codeUrl, null);
            }
            return new UnifiedOrderResult(false, null, null, "微信下单失败: " + resp.getStatusCode());
        } catch (Exception e) {
            log.error("微信统一下单失败: orderNo={}", orderNo, e);
            return new UnifiedOrderResult(false, null, null, e.getMessage());
        }
    }

    @Override
    public CallbackVerifyResult verifyCallback(Map<String, String> params, String rawBody) {
        if (!isEnabled()) {
            String orderNo = params.get("out_trade_no");
            String amountStr = params.get("total");
            long amount = amountStr != null ? Long.parseLong(amountStr) : 0L;
            String tradeNo = params.get("transaction_id");
            return new CallbackVerifyResult(true, orderNo, amount, tradeNo, null);
        }
        String signature = params.get("signature");
        if (signature == null) {
            return new CallbackVerifyResult(false, null, 0L, null, "缺少签名");
        }
        String orderNo = params.get("out_trade_no");
        String tradeNo = params.get("transaction_id");
        String amountStr = params.get("amount");
        long amount = amountStr != null ? Long.parseLong(amountStr) : 0L;
        String expectedSig = sign(params, paymentProperties.getWechat().getApiKey());
        if (!signature.equals(expectedSig)) {
            return new CallbackVerifyResult(false, orderNo, amount, tradeNo, "签名校验失败");
        }
        return new CallbackVerifyResult(true, orderNo, amount, tradeNo, null);
    }

    @Override
    public boolean refund(String orderNo, String refundNo, long totalAmountFen, long refundAmountFen, String reason) {
        if (!isEnabled()) {
            log.info("微信退款(mock): orderNo={}, refundNo={}, amount={}", orderNo, refundNo, refundAmountFen);
            return true;
        }
        try {
            PaymentProperties.ChannelConfig cfg = paymentProperties.getWechat();
            String url = cfg.getBaseUrl() + REFUND_PATH;
            Map<String, Object> body = Map.of(
                    "out_trade_no", orderNo,
                    "out_refund_no", refundNo,
                    "reason", reason != null ? reason : "",
                    "amount", Map.of(
                            "refund", refundAmountFen,
                            "total", totalAmountFen,
                            "currency", "CNY"
                    ),
                    "notify_url", cfg.getRefundNotifyUrl()
            );
            ResponseEntity<Map> resp = restTemplate.postForEntity(url, body, Map.class);
            return resp.getStatusCode().is2xxSuccessful();
        } catch (Exception e) {
            log.error("微信退款失败: orderNo={}, refundNo={}", orderNo, refundNo, e);
            return false;
        }
    }

    @Override
    public boolean autoDeduct(String orderNo, long amountFen, String description, String payToken) {
        log.warn("微信代扣需签约协议，本次走统一下单+用户主动支付流程: orderNo={}", orderNo);
        UnifiedOrderResult result = unifiedOrder(orderNo, amountFen, description, Map.of());
        return result.success();
    }

    @Override
    public boolean closeOrder(String orderNo) {
        if (!isEnabled()) {
            log.info("微信关单(mock): orderNo={}", orderNo);
            return true;
        }
        try {
            PaymentProperties.ChannelConfig cfg = paymentProperties.getWechat();
            String url = cfg.getBaseUrl() + String.format(CLOSE_ORDER_PATH, orderNo);
            Map<String, Object> body = Map.of("mchid", cfg.getMchId());
            ResponseEntity<Map> resp = restTemplate.postForEntity(url, body, Map.class);
            return resp.getStatusCode().is2xxSuccessful();
        } catch (Exception e) {
            log.error("微信关单失败: orderNo={}", orderNo, e);
            return false;
        }
    }

    private UnifiedOrderResult mockOrder(String orderNo) {
        String qr = "weixin://wxpay/bizpayurl?pr=" + orderNo;
        return new UnifiedOrderResult(true, null, qr, null);
    }

    private String sign(Map<String, String> params, String apiKey) {
        TreeMap<String, String> sorted = new TreeMap<>(params);
        sorted.remove("sign");
        StringBuilder sb = new StringBuilder();
        sorted.forEach((k, v) -> {
            if (v != null && !v.isEmpty()) {
                if (!sb.isEmpty()) {
                    sb.append("&");
                }
                sb.append(k).append("=").append(v);
            }
        });
        sb.append("&key=").append(apiKey);
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] digest = md.digest(sb.toString().getBytes(StandardCharsets.UTF_8));
            StringBuilder hex = new StringBuilder();
            for (byte b : digest) {
                hex.append(String.format("%02x", b));
            }
            return hex.toString();
        } catch (Exception e) {
            return "";
        }
    }
}
