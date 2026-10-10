package com.pei.dehaze.controller;

import static org.hamcrest.Matchers.containsString;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.content;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.model.vo.AiBalanceVO;
import com.pei.dehaze.model.vo.AiBillingRecordVO;
import com.pei.dehaze.model.vo.AiModelCostVO;
import com.pei.dehaze.service.AiBillingService;
import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;

/**
 * AI 计费余额/成本单价端点金额表示契约。
 *
 * <p>python（唯一行为事实源）中 {@code BalanceResult.credits_balance} 与
 * {@code ModelCostDetailResult.unit_price} 均为 {@code Decimal}，序列化为 <b>JSON 字符串</b>；
 * java 侧 {@code AiBalanceVO.creditsBalance}、{@code AiModelCostVO.Detail.unitPrice} 必须带
 * {@code @JsonSerialize(using = ToStringSerializer.class)} 对齐同一口径（裸 {@code BigDecimal}
 * 会下发 JSON 数字，跨端类型不一致）。本用例按「对外 camelCase 键名 + 字符串类型」双重断言，
 * 去掉注解后必然转红（摘除反证）。
 */
@DisplayName("AiBillingController 余额/成本单价金额序列化契约")
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
class AiBillingControllerAmountContractTest {

    @Mock
    private AiBillingService billingService;

    private MockMvc mockMvc;

    @BeforeEach
    void setUp() {
        mockMvc = MockMvcBuilders.standaloneSetup(new AiBillingController(billingService))
                .build();
    }

    @Test
    @DisplayName("余额查询：creditsBalance 下发 JSON 字符串（对齐 python Decimal）")
    void balanceCreditsIsJsonString() throws Exception {
        when(billingService.getBalance(any())).thenReturn(balance("100.00"));

        mockMvc.perform(get("/api/v1/ai-billing/balance"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"))
                .andExpect(jsonPath("$.data.creditsBalance").isString())
                .andExpect(jsonPath("$.data.creditsBalance").value("100.00"))
                // 键名为 camelCase，且不得以 JSON 数字形态出现
                .andExpect(content().string(containsString("\"creditsBalance\":\"100.00\"")));
    }

    @Test
    @DisplayName("积分调整：返回的 creditsBalance 同样按字符串下发")
    void adjustCreditsIsJsonString() throws Exception {
        when(billingService.adjustCredits(any())).thenReturn(balance("150.00"));

        mockMvc.perform(post("/api/v1/ai-billing/adjust")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"userId\":9,\"amount\":50,\"reason\":\"测试调整\"}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.data.creditsBalance").isString())
                .andExpect(jsonPath("$.data.creditsBalance").value("150.00"))
                .andExpect(content().string(containsString("\"creditsBalance\":\"150.00\"")));
    }

    @Test
    @DisplayName("成本单价列表：details[].unitPrice 下发 JSON 字符串（对齐 python Decimal）")
    void costUnitPriceIsJsonString() throws Exception {
        when(billingService.listCosts(any())).thenReturn(costPage("1.0000"));

        mockMvc.perform(get("/api/v1/ai-billing/costs"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.data.list[0].details[0].unitPrice").isString())
                .andExpect(jsonPath("$.data.list[0].details[0].unitPrice").value("1.0000"))
                .andExpect(content().string(containsString("\"unitPrice\":\"1.0000\"")));
    }

    @Test
    @DisplayName("计费明细：quantity 下发 JSON 字符串（对齐 python Decimal，勿退回裸 BigDecimal）")
    void recordQuantityIsJsonString() throws Exception {
        when(billingService.listRecords(any())).thenReturn(recordPage("12.3400", "ten_thousand_chars"));

        mockMvc.perform(get("/api/v1/ai-billing/records"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.data.list[0].quantity").isString())
                .andExpect(jsonPath("$.data.list[0].quantity").value("12.3400"))
                .andExpect(jsonPath("$.data.list[0].quantityUnit").value("ten_thousand_chars"))
                .andExpect(content().string(containsString("\"quantity\":\"12.3400\"")));
    }

    private static Page<AiBillingRecordVO> recordPage(String quantity, String quantityUnit) {
        AiBillingRecordVO vo = new AiBillingRecordVO();
        vo.setId(1L);
        vo.setUserId(9L);
        vo.setModel("huayan");
        vo.setBillType("tts");
        vo.setInputTokens(0);
        vo.setCachedInputTokens(0);
        vo.setOutputTokens(0);
        vo.setQuantity(quantity == null ? null : new BigDecimal(quantity));
        vo.setQuantityUnit(quantityUnit);
        vo.setCredits(3);
        vo.setCreditsSaved(0);
        vo.setQuotaConsumed(0);
        vo.setPreDeduct(0);
        vo.setCreateTime(LocalDateTime.of(2026, 7, 1, 0, 0, 0));

        Page<AiBillingRecordVO> page = new Page<>(1, 20, 1);
        page.setRecords(List.of(vo));
        return page;
    }

    private static AiBalanceVO balance(String credits) {
        AiBalanceVO vo = new AiBalanceVO();
        vo.setUserId(9L);
        vo.setCreditsBalance(new BigDecimal(credits));
        vo.setArrearsStatus(false);
        vo.setDailyUsed(0);
        vo.setDailyLimit(0);
        vo.setMonthlyUsed(0);
        vo.setMonthlyLimit(0);
        return vo;
    }

    private static Page<AiModelCostVO> costPage(String unitPrice) {
        AiModelCostVO vo = new AiModelCostVO();
        vo.setId(1L);
        vo.setModelId("test_model_cost");
        vo.setProviderId(1L);
        vo.setPriceVersion(1);
        vo.setCurrency("CNY");
        vo.setStatus(1);

        AiModelCostVO.Detail detail = new AiModelCostVO.Detail();
        detail.setId(1L);
        detail.setPriceId(1L);
        detail.setTokenType("input");
        detail.setTimeSlot("peak");
        detail.setMinTokens(0L);
        detail.setUnitPrice(new BigDecimal(unitPrice));
        vo.setDetails(List.of(detail));

        Page<AiModelCostVO> page = new Page<>(1, 20, 1);
        page.setRecords(List.of(vo));
        return page;
    }
}
