package com.pei.dehaze.controller;

import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.service.OrderService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.servlet.http.HttpServletRequest;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;

@Slf4j
@Tag(name = "14.支付回调")
@RestController
@RequestMapping("/api/v1/payments")
@RequiredArgsConstructor
public class PaymentCallbackController {

    private final OrderService orderService;

    @Operation(summary = "微信支付回调")
    @PostMapping("/wechat/callback")
    public Map<String, Object> wechatCallback(HttpServletRequest request) {
        Map<String, String> params = toParamMap(request);
        String rawBody = readBody(request);
        log.info("收到微信支付回调: orderNo={}", params.get("out_trade_no"));
        boolean ok = orderService.handlePaymentCallback("wechat", params, rawBody);
        Map<String, Object> resp = new HashMap<>();
        resp.put("code", ok ? "SUCCESS" : "FAIL");
        resp.put("message", ok ? "成功" : "失败");
        return resp;
    }

    @Operation(summary = "支付宝支付回调")
    @PostMapping("/alipay/callback")
    public String alipayCallback(HttpServletRequest request) {
        Map<String, String> params = toParamMap(request);
        String rawBody = readBody(request);
        log.info("收到支付宝支付回调: orderNo={}", params.get("out_trade_no"));
        boolean ok = orderService.handlePaymentCallback("alipay", params, rawBody);
        return ok ? "success" : "fail";
    }

    private Map<String, String> toParamMap(HttpServletRequest request) {
        Map<String, String> params = new HashMap<>();
        request.getParameterMap().forEach((k, v) -> {
            if (v != null && v.length > 0) {
                params.put(k, v[0]);
            }
        });
        return params;
    }

    private String readBody(HttpServletRequest request) {
        try {
            return new String(request.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
        } catch (Exception e) {
            return "";
        }
    }
}
