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
public class AlipayService implements PaymentChannelService {

    private static final String CHANNEL_TYPE = "alipay";
    private static final String PRECREATE_PATH = "/gateway.do?method=alipay.trade.precreate";
    private static final String REFUND_PATH = "/gateway.do?method=alipay.trade.refund";
    private static final String CLOSE_PATH = "/gateway.do?method=alipay.trade.close";

    private final PaymentProperties paymentProperties;
    @Qualifier("paymentRestTemplate")
    private final RestTemplate restTemplate;

    @Override
    public String getChannelType() {
        return CHANNEL_TYPE;
    }

    @Override
    public boolean isEnabled() {
        return paymentProperties.getAlipay().isEnabled();
    }

    @Override
    public UnifiedOrderResult unifiedOrder(String orderNo, long amountFen, String description, Map<String, String> extra) {
        if (!isEnabled()) {
            return mockOrder(orderNo);
        }
        try {
            PaymentProperties.ChannelConfig cfg = paymentProperties.getAlipay();
            String url = cfg.getBaseUrl() + PRECREATE_PATH;
            Map<String, Object> bizContent = Map.of(
                    "out_trade_no", orderNo,
                    "total_amount", formatYuan(amountFen),
                    "subject", description
            );
            Map<String, Object> body = Map.of(
                    "app_id", cfg.getAppId(),
                    "method", "alipay.trade.precreate",
                    "biz_content", bizContent,
                    "notify_url", cfg.getNotifyUrl()
            );
            ResponseEntity<Map> resp = restTemplate.postForEntity(url, body, Map.class);
            if (resp.getStatusCode().is2xxSuccessful() && resp.getBody() != null) {
                String qrCode = (String) resp.getBody().get("qr_code");
                return new UnifiedOrderResult(true, null, qrCode, null);
            }
            return new UnifiedOrderResult(false, null, null, "支付宝下单失败: " + resp.getStatusCode());
        } catch (Exception e) {
            log.error("支付宝统一下单失败: orderNo={}", orderNo, e);
            return new UnifiedOrderResult(false, null, null, e.getMessage());
        }
    }

    @Override
    public CallbackVerifyResult verifyCallback(Map<String, String> params, String rawBody) {
        if (!isEnabled()) {
            String orderNo = params.get("out_trade_no");
            String amountStr = params.get("total_amount");
            long amount = amountStr != null ? parseFen(amountStr) : 0L;
            String tradeNo = params.get("trade_no");
            return new CallbackVerifyResult(true, orderNo, amount, tradeNo, null);
        }
        String sign = params.get("sign");
        if (sign == null) {
            return new CallbackVerifyResult(false, null, 0L, null, "缺少签名");
        }
        String orderNo = params.get("out_trade_no");
        String tradeNo = params.get("trade_no");
        String amountStr = params.get("total_amount");
        long amount = amountStr != null ? parseFen(amountStr) : 0L;
        String expectedSign = sign(params, paymentProperties.getAlipay().getApiKey());
        if (!sign.equals(expectedSign)) {
            return new CallbackVerifyResult(false, orderNo, amount, tradeNo, "签名校验失败");
        }
        return new CallbackVerifyResult(true, orderNo, amount, tradeNo, null);
    }

    @Override
    public boolean refund(String orderNo, String refundNo, long totalAmountFen, long refundAmountFen, String reason) {
        if (!isEnabled()) {
            log.debug("支付宝退款(mock): orderNo={}, refundNo={}, amount={}", orderNo, refundNo, refundAmountFen);
            return true;
        }
        try {
            PaymentProperties.ChannelConfig cfg = paymentProperties.getAlipay();
            String url = cfg.getBaseUrl() + REFUND_PATH;
            Map<String, Object> bizContent = Map.of(
                    "out_trade_no", orderNo,
                    "out_request_no", refundNo,
                    "refund_amount", formatYuan(refundAmountFen),
                    "refund_reason", reason != null ? reason : ""
            );
            Map<String, Object> body = Map.of(
                    "app_id", cfg.getAppId(),
                    "method", "alipay.trade.refund",
                    "biz_content", bizContent
            );
            ResponseEntity<Map> resp = restTemplate.postForEntity(url, body, Map.class);
            return resp.getStatusCode().is2xxSuccessful();
        } catch (Exception e) {
            log.error("支付宝退款失败: orderNo={}, refundNo={}", orderNo, refundNo, e);
            return false;
        }
    }

    @Override
    public boolean autoDeduct(String orderNo, long amountFen, String description, String payToken) {
        log.warn("支付宝代扣需签约协议，本次走预下单+用户主动支付流程: orderNo={}", orderNo);
        UnifiedOrderResult result = unifiedOrder(orderNo, amountFen, description, Map.of());
        return result.success();
    }

    @Override
    public boolean closeOrder(String orderNo) {
        if (!isEnabled()) {
            log.debug("支付宝关单(mock): orderNo={}", orderNo);
            return true;
        }
        try {
            PaymentProperties.ChannelConfig cfg = paymentProperties.getAlipay();
            String url = cfg.getBaseUrl() + CLOSE_PATH;
            Map<String, Object> bizContent = Map.of(
                    "out_trade_no", orderNo
            );
            Map<String, Object> body = Map.of(
                    "app_id", cfg.getAppId(),
                    "method", "alipay.trade.close",
                    "biz_content", bizContent
            );
            ResponseEntity<Map> resp = restTemplate.postForEntity(url, body, Map.class);
            return resp.getStatusCode().is2xxSuccessful();
        } catch (Exception e) {
            log.error("支付宝关单失败: orderNo={}", orderNo, e);
            return false;
        }
    }

    private UnifiedOrderResult mockOrder(String orderNo) {
        String qr = "https://qr.alipay.com/" + orderNo;
        return new UnifiedOrderResult(true, qr, qr, null);
    }

    private String formatYuan(long amountFen) {
        return String.format("%.2f", amountFen / 100.0);
    }

    private long parseFen(String yuan) {
        try {
            return Math.round(Double.parseDouble(yuan) * 100);
        } catch (NumberFormatException e) {
            return 0L;
        }
    }

    private String sign(Map<String, String> params, String apiKey) {
        TreeMap<String, String> sorted = new TreeMap<>(params);
        sorted.remove("sign");
        sorted.remove("sign_type");
        StringBuilder sb = new StringBuilder();
        sorted.forEach((k, v) -> {
            if (v != null && !v.isEmpty()) {
                if (!sb.isEmpty()) {
                    sb.append("&");
                }
                sb.append(k).append("=").append(v);
            }
        });
        sb.append(apiKey);
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
