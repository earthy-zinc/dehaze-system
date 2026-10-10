package com.pei.dehaze.controller;

import static org.hamcrest.Matchers.containsString;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.content;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.pei.dehaze.model.vo.AiBillVO;
import com.pei.dehaze.service.AiBillingService;
import java.math.BigDecimal;
import java.util.Map;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;

/**
 * 月结账单端点金额表示契约。
 *
 * <p>python（唯一行为事实源）的 {@code BillResult.balance_start/balance_end} 是 {@code Decimal}，
 * {@code model_dump(mode="json")} 序列化为 <b>JSON 字符串</b>；java 的 {@code AiBillVO} 必须用
 * {@code ToStringSerializer} 对齐同一口径（裸 {@code BigDecimal} 会下发 JSON 数字，跨端类型不一致），
 * 故此处按「camelCase 对外契约 + 字符串类型」双重断言，含负数（欠费）与零值边界。
 */
@DisplayName("AiBillingController 月结账单金额序列化契约")
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
class AiBillingControllerBillContractTest {

    @Mock
    private AiBillingService billingService;

    private MockMvc mockMvc;

    @BeforeEach
    void setUp() {
        mockMvc = MockMvcBuilders.standaloneSetup(new AiBillingController(billingService))
                .build();
    }

    @Test
    @DisplayName("账单查询：balanceStart/balanceEnd 下发 JSON 字符串（对齐 python Decimal）")
    void billBalancesAreJsonStrings() throws Exception {
        when(billingService.getBill("2026-07")).thenReturn(bill("3000.00", "1500.00"));

        mockMvc.perform(get("/api/v1/ai-billing/bills/2026-07"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"))
                .andExpect(jsonPath("$.data.balanceStart").isString())
                .andExpect(jsonPath("$.data.balanceStart").value("3000.00"))
                .andExpect(jsonPath("$.data.balanceEnd").isString())
                .andExpect(jsonPath("$.data.balanceEnd").value("1500.00"))
                // 键名仍为 camelCase，且金额不得以 JSON 数字形态出现
                .andExpect(content().string(containsString("\"balanceStart\":\"3000.00\"")))
                .andExpect(content().string(containsString("\"balanceEnd\":\"1500.00\"")));
    }

    @Test
    @DisplayName("账单查询：0 与欠费负值同样按字符串下发（不出现裸数字）")
    void billZeroAndNegativeBalancesAreJsonStrings() throws Exception {
        when(billingService.getBill("2026-08")).thenReturn(bill("0", "-25.50"));

        mockMvc.perform(get("/api/v1/ai-billing/bills/2026-08"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.data.balanceStart").isString())
                .andExpect(jsonPath("$.data.balanceStart").value("0"))
                .andExpect(jsonPath("$.data.balanceEnd").isString())
                .andExpect(jsonPath("$.data.balanceEnd").value("-25.50"))
                .andExpect(content().string(containsString("\"balanceEnd\":\"-25.50\"")));
    }

    @Test
    @DisplayName("账单下载：与账单查询同一序列化口径")
    void downloadBillBalancesAreJsonStrings() throws Exception {
        when(billingService.getBill("2026-07")).thenReturn(bill("3000.00", "1500.00"));

        mockMvc.perform(get("/api/v1/ai-billing/bills/2026-07/download"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.data.balanceStart").isString())
                .andExpect(jsonPath("$.data.balanceStart").value("3000.00"));
    }

    private static AiBillVO bill(String balanceStart, String balanceEnd) {
        AiBillVO vo = new AiBillVO();
        vo.setUserId(1L);
        vo.setMonth("2026-07");
        vo.setTotalConsume(100);
        vo.setTotalRecharge(0);
        vo.setTotalRefund(0);
        vo.setBalanceStart(new BigDecimal(balanceStart));
        vo.setBalanceEnd(new BigDecimal(balanceEnd));
        vo.setItemSummary(Map.of("chat", 100));
        return vo;
    }
}
