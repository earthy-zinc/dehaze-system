package com.pei.dehaze.service.payment;

import java.util.Map;

public interface PaymentChannelService {

    String getChannelType();

    boolean isEnabled();

    UnifiedOrderResult unifiedOrder(String orderNo, long amountFen, String description, Map<String, String> extra);

    CallbackVerifyResult verifyCallback(Map<String, String> params, String rawBody);

    boolean refund(String orderNo, String refundNo, long totalAmountFen, long refundAmountFen, String reason);

    boolean closeOrder(String orderNo);

    boolean autoDeduct(String orderNo, long amountFen, String description, String payToken);

    record UnifiedOrderResult(boolean success, String payUrl, String qrCode, String errorMessage) {
    }

    record CallbackVerifyResult(boolean success, String orderNo, long amountFen, String channelTradeNo, String errorMessage) {
    }
}
