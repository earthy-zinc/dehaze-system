package com.pei.dehaze.model.form;

import com.pei.dehaze.model.query.SkillPageQuery;
import jakarta.validation.ConstraintViolation;
import jakarta.validation.Validation;
import jakarta.validation.Validator;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.util.List;
import java.util.Set;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * 表单/查询约束与 python 侧 parity 校验（字段级）。
 * <p>
 * python 是事实源：`models/schema` 的 `ge/le/gt/min_length/max_length` 必须在 java DTO 上有对应注解，
 * 否则越界值静默入库（或超长串打到 DB 列宽变成 B0001/500）。仅覆盖已确认存在分叉的字段；
 * python 无界字段（如各表单 `status`、MCP 列表 `status`、价格 `max_tokens`）刻意不加约束。
 */
@DisplayName("AI 域表单约束 parity 校验（对齐 python schema）")
class AiFormValidationParityTest {

    private static Validator validator;

    @BeforeAll
    static void setUpValidator() {
        validator = Validation.buildDefaultValidatorFactory().getValidator();
    }

    @Test
    @DisplayName("KbUpdateForm：name/searchStrategy/hybridWeight/topK/scoreThreshold/rerankModel/embeddingModel/chunkingStrategy 越界被拒")
    void kbUpdateFormRejectsOutOfRangeValues() {
        assertThat(violations(validKbUpdateForm())).isEmpty();

        KbUpdateForm form = validKbUpdateForm();
        form.setName("");
        form.setSearchStrategy("bogus");
        form.setHybridWeight(new BigDecimal("1.5"));
        form.setTopK(101);
        // python 侧 score_threshold 是 lt=1：等于上界必须被拒（左闭右开最易误写成 le）
        form.setScoreThreshold(BigDecimal.ONE);
        form.setRerankModel("x".repeat(65));
        form.setEmbeddingModel("x".repeat(65));
        // 不可修改字段的非法字面量在 python 属请求校验阶段（A0400），非服务层的 A0500"不可修改"业务错误
        form.setChunkingStrategy("bogus");

        Set<String> properties = violationProperties(form);
        assertThat(properties).contains("name", "searchStrategy", "hybridWeight", "topK",
                "scoreThreshold", "rerankModel", "embeddingModel", "chunkingStrategy");

        KbUpdateForm topKLow = validKbUpdateForm();
        topKLow.setTopK(0);
        assertThat(violationProperties(topKLow)).contains("topK");

        KbUpdateForm weightLow = validKbUpdateForm();
        weightLow.setHybridWeight(new BigDecimal("-0.1"));
        assertThat(violationProperties(weightLow)).contains("hybridWeight");

        // python 侧 search_strategy 是 Literal[vector|keyword|hybrid]，合法值与 null 都放行
        KbUpdateForm validStrategy = validKbUpdateForm();
        validStrategy.setSearchStrategy("keyword");
        assertThat(violationProperties(validStrategy)).doesNotContain("searchStrategy");
    }

    @Test
    @DisplayName("SkillPageQuery：status 仅允许 0/1，null 放行")
    void skillPageQueryRejectsStatusOutOfRange() {
        assertThat(violations(new SkillPageQuery())).isEmpty();

        SkillPageQuery tooHigh = new SkillPageQuery();
        tooHigh.setStatus(2);
        assertThat(violationProperties(tooHigh)).contains("status");

        SkillPageQuery negative = new SkillPageQuery();
        negative.setStatus(-1);
        assertThat(violationProperties(negative)).contains("status");
    }

    @Test
    @DisplayName("ProviderKey 表单：dailyQuota≥1、rpmLimit≥0、name≤128，null 放行")
    void providerKeyFormsRejectOutOfRangeQuotas() {
        ProviderKeyForm create = new ProviderKeyForm();
        create.setName("key");
        create.setKey("plain");
        assertThat(violations(create)).isEmpty();

        create.setName("x".repeat(129));
        create.setDailyQuota(0);
        create.setRpmLimit(-1);
        assertThat(violationProperties(create)).contains("name", "dailyQuota", "rpmLimit");

        ProviderKeyUpdateForm update = new ProviderKeyUpdateForm();
        assertThat(violations(update)).isEmpty();

        update.setDailyQuota(0);
        update.setRpmLimit(-1);
        update.setName("");
        assertThat(violationProperties(update)).contains("dailyQuota", "rpmLimit", "name");
    }

    @Test
    @DisplayName("ModelPrice 表单：modelId≤64、unit≤24、minTokens≥0、unitPrice≥0；maxTokens 无界不动")
    void modelPriceFormsRejectOutOfRangeValues() {
        ModelPriceForm form = new ModelPriceForm();
        form.setModelId("m");
        form.setProviderId(1L);
        form.getDetails().add(new ModelPriceForm.Detail());
        assertThat(violations(form)).isEmpty();

        form.setModelId("x".repeat(65));
        form.setUnit("x".repeat(25));
        ModelPriceForm.Detail detail = form.getDetails().get(0);
        detail.setMinTokens(-1L);
        detail.setUnitPrice(new BigDecimal("-0.01"));
        detail.setMaxTokens(-999L); // python max_tokens 无界 → 不产生违规

        assertThat(violationProperties(form)).contains("modelId", "unit");
        assertThat(violations(detail).stream().map(v -> v.getPropertyPath().toString()))
                .contains("minTokens", "unitPrice")
                .doesNotContain("maxTokens");

        ModelPriceUpdateForm update = new ModelPriceUpdateForm();
        assertThat(violations(update)).isEmpty();
        update.setUnit("x".repeat(25));
        assertThat(violationProperties(update)).contains("unit");
    }

    @Test
    @DisplayName("AiModel 表单：dimension≥1、maxContextTokens/maxOutputTokens≥1、promptCachePrefixLen≥0、长度界")
    void aiModelFormsRejectOutOfRangeValues() {
        AiModelForm create = new AiModelForm();
        create.setProviderId(1L);
        create.setModelId("m");
        create.setDisplayName("模型");
        assertThat(violations(create)).isEmpty();

        create.setModelId("x".repeat(65));
        create.setDimension(0L);
        create.setDisplayName("x".repeat(129));
        create.setMaxContextTokens(0);
        create.setMaxOutputTokens(0);
        create.setPromptCachePrefixLen(-1);
        assertThat(violationProperties(create))
                .contains("modelId", "dimension", "displayName", "maxContextTokens",
                        "maxOutputTokens", "promptCachePrefixLen");

        AiModelUpdateForm update = new AiModelUpdateForm();
        assertThat(violations(update)).isEmpty();

        update.setDimension(0L);
        update.setDisplayName("");
        update.setMaxContextTokens(0);
        update.setMaxOutputTokens(0);
        update.setPromptCachePrefixLen(-1);
        assertThat(violationProperties(update))
                .contains("dimension", "displayName", "maxContextTokens", "maxOutputTokens",
                        "promptCachePrefixLen");
    }

    @Test
    @DisplayName("MCP 表单：name 1..128、description/endpoint ≤512、authType ≤32、apiKey ≤1024")
    void mcpFormsRejectOutOfRangeValues() {
        McpServerForm create = new McpServerForm();
        create.setName("server");
        assertThat(violations(create)).isEmpty();

        create.setName("x".repeat(129));
        create.setDescription("x".repeat(513));
        create.setEndpoint("x".repeat(513));
        create.setAuthType("x".repeat(33));
        assertThat(violationProperties(create))
                .contains("name", "description", "endpoint", "authType");

        McpServerUpdateForm update = new McpServerUpdateForm();
        assertThat(violations(update)).isEmpty();
        update.setName("");
        update.setDescription("x".repeat(513));
        assertThat(violationProperties(update)).contains("name", "description");

        McpCredentialForm credential = new McpCredentialForm();
        credential.setApiKey("x".repeat(1025));
        assertThat(violationProperties(credential)).contains("apiKey");
    }

    @Test
    @DisplayName("ProviderForm：providerCode 1..32、displayName 1..128、apiBaseUrl 1..512、protocolType/authType ≤32")
    void providerFormRejectsOutOfRangeValues() {
        ProviderForm form = new ProviderForm();
        form.setProviderCode("openai");
        form.setDisplayName("OpenAI");
        form.setApiBaseUrl("https://api.openai.com");
        assertThat(violations(form)).isEmpty();

        form.setProviderCode("x".repeat(33));
        form.setDisplayName("");
        form.setApiBaseUrl("x".repeat(513));
        form.setProtocolType("x".repeat(33));
        form.setAuthType("x".repeat(33));
        assertThat(violationProperties(form))
                .contains("providerCode", "displayName", "apiBaseUrl", "protocolType", "authType");

        ProviderUpdateForm update = new ProviderUpdateForm();
        assertThat(violations(update)).isEmpty();

        update.setDisplayName("");
        update.setApiBaseUrl("x".repeat(513));
        update.setProtocolType("x".repeat(33));
        update.setAuthType("x".repeat(33));
        assertThat(violationProperties(update))
                .contains("displayName", "apiBaseUrl", "protocolType", "authType");
    }

    @Test
    @DisplayName("TestSetForm：question 1..1000、expectedChunkIds 非空")
    void testSetFormRejectsOutOfRangeValues() {
        TestSetForm form = new TestSetForm();
        form.setQuestion("问题");
        form.setExpectedChunkIds(List.of(1L));
        assertThat(violations(form)).isEmpty();

        form.setQuestion("x".repeat(1001));
        form.setExpectedChunkIds(List.of());
        assertThat(violationProperties(form)).contains("question", "expectedChunkIds");
    }

    @Test
    @DisplayName("SkillUpdateForm：name/description/instruction 非空且含上界、scene ≤255，null 放行")
    void skillUpdateFormRejectsOutOfRangeValues() {
        SkillUpdateForm form = new SkillUpdateForm();
        assertThat(violations(form)).isEmpty();

        form.setName("");
        form.setDescription("x".repeat(501));
        form.setInstruction("");
        form.setScene("x".repeat(256));
        assertThat(violationProperties(form))
                .contains("name", "description", "instruction", "scene");

        SkillUpdateForm tooLongName = new SkillUpdateForm();
        tooLongName.setName("x".repeat(129));
        assertThat(violationProperties(tooLongName)).contains("name");
    }

    private static KbUpdateForm validKbUpdateForm() {
        KbUpdateForm form = new KbUpdateForm();
        form.setName("知识库");
        form.setHybridWeight(new BigDecimal("0.7"));
        form.setTopK(5);
        form.setScoreThreshold(new BigDecimal("0.5"));
        return form;
    }

    private static <T> Set<ConstraintViolation<T>> violations(T bean) {
        return validator.validate(bean);
    }

    private static <T> Set<String> violationProperties(T bean) {
        return violations(bean).stream()
                .map(v -> v.getPropertyPath().toString())
                .collect(java.util.stream.Collectors.toSet());
    }
}
