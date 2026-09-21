package com.pei.dehaze.mapper;

import com.pei.dehaze.model.read.AiBillTypeRead;
import com.pei.dehaze.model.read.AiBillingModelRead;
import com.pei.dehaze.model.read.AiBillingPeriodRead;
import com.pei.dehaze.model.read.AiBillingStatRead;
import com.pei.dehaze.model.read.AiCostStatRead;
import com.pei.dehaze.model.read.AiCreditSourceRead;
import com.pei.dehaze.model.read.AiOrderIncomeRead;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;

/**
 * AI 计费域聚合查询（统计/账单/成本-利润），不承担写入。
 *
 * <p>token 相关列仅基于 chat 类记录（asr/tts 的 input_tokens 存秒数/字符数，与 token
 * 计量口径不同），与 python {@code ai_billing_repository.CHAT_BILL_TYPES} 口径一致。
 */
@Mapper
public interface AiBillingInsightMapper {

    /**
     * 按维度聚合计费统计（管理员）：user/model/billType/day
     */
    @Select("<script>SELECT " +
            "<choose>" +
            "<when test=\"groupBy == 'user'\">user_id</when>" +
            "<when test=\"groupBy == 'model'\">model</when>" +
            "<when test=\"groupBy == 'billType'\">bill_type</when>" +
            "<otherwise>DATE_FORMAT(create_time, '%Y-%m-%d')</otherwise>" +
            "</choose> AS dimension, " +
            "COALESCE(SUM(credits), 0) AS totalCredits, " +
            "COALESCE(SUM(CASE WHEN bill_type IN ('chat', 'chat_subagent') THEN input_tokens ELSE 0 END), 0) AS totalInputTokens, " +
            "COALESCE(SUM(CASE WHEN bill_type IN ('chat', 'chat_subagent') THEN output_tokens ELSE 0 END), 0) AS totalOutputTokens, " +
            "COALESCE(SUM(CASE WHEN bill_type IN ('chat', 'chat_subagent') THEN cached_input_tokens ELSE 0 END), 0) AS chatCachedTokens, " +
            "COALESCE(SUM(credits_saved), 0) AS creditsSaved, " +
            "COALESCE(SUM(CASE WHEN actual_model IS NOT NULL THEN 1 ELSE 0 END), 0) AS degradationCount " +
            "FROM sys_ai_billing WHERE 1 = 1 " +
            "<if test='userId != null'> AND user_id = #{userId}</if>" +
            "<if test='modelId != null'> AND model = #{modelId}</if>" +
            "<if test='billType != null'> AND bill_type = #{billType}</if>" +
            "<if test='start != null'> AND create_time &gt;= #{start}</if>" +
            "<if test='end != null'> AND create_time &lt;= #{end}</if>" +
            " GROUP BY dimension ORDER BY dimension</script>")
    List<AiBillingStatRead> statsByDimension(@Param("groupBy") String groupBy,
                                             @Param("userId") Long userId,
                                             @Param("modelId") String modelId,
                                             @Param("billType") String billType,
                                             @Param("start") LocalDateTime start,
                                             @Param("end") LocalDateTime end);

    /**
     * 用户消耗趋势（按日/月聚合，仅 chat 类记录）
     */
    @Select("SELECT DATE_FORMAT(create_time, #{fmt}) AS date, " +
            "COALESCE(SUM(credits), 0) AS credits, " +
            "COALESCE(SUM(input_tokens), 0) AS inputTokens, " +
            "COALESCE(SUM(output_tokens), 0) AS outputTokens, " +
            "COALESCE(SUM(credits_saved), 0) AS creditsSaved, " +
            "COALESCE(SUM(cached_input_tokens), 0) AS cachedInputTokens " +
            "FROM sys_ai_billing WHERE user_id = #{userId} AND bill_type IN ('chat', 'chat_subagent') " +
            "AND create_time >= #{start} AND create_time <= #{end} " +
            "GROUP BY date ORDER BY date")
    List<AiBillingPeriodRead> sumGroupByPeriod(@Param("userId") Long userId,
                                               @Param("start") LocalDateTime start,
                                               @Param("end") LocalDateTime end,
                                               @Param("fmt") String fmt);

    /**
     * 用户模型消耗分布（仅 chat 类记录）
     */
    @Select("SELECT model, COALESCE(SUM(credits), 0) AS credits, " +
            "COALESCE(SUM(input_tokens), 0) AS inputTokens, COALESCE(SUM(output_tokens), 0) AS outputTokens " +
            "FROM sys_ai_billing WHERE user_id = #{userId} AND bill_type IN ('chat', 'chat_subagent') " +
            "AND create_time >= #{start} AND create_time <= #{end} GROUP BY model")
    List<AiBillingModelRead> sumGroupByModel(@Param("userId") Long userId,
                                             @Param("start") LocalDateTime start,
                                             @Param("end") LocalDateTime end);

    /**
     * 账单按计费类型汇总
     */
    @Select("SELECT bill_type AS billType, COALESCE(SUM(credits), 0) AS credits FROM sys_ai_billing " +
            "WHERE user_id = #{userId} AND create_time >= #{start} AND create_time <= #{end} " +
            "GROUP BY bill_type")
    List<AiBillTypeRead> sumByBillType(@Param("userId") Long userId,
                                       @Param("start") LocalDateTime start,
                                       @Param("end") LocalDateTime end);

    /**
     * 成本合计（元；cost 未回填的记录为 NULL，由 COALESCE 归一）
     */
    @Select("<script>SELECT COALESCE(SUM(cost), 0) FROM sys_ai_billing WHERE 1 = 1 " +
            "<if test='start != null'> AND create_time &gt;= #{start}</if>" +
            "<if test='end != null'> AND create_time &lt;= #{end}</if></script>")
    BigDecimal sumCost(@Param("start") LocalDateTime start, @Param("end") LocalDateTime end);

    /**
     * 按模型/供应商分解成本（仅统计已回填成本的记录）
     */
    @Select("<script>SELECT " +
            "<choose><when test=\"dimension == 'model'\">model</when><otherwise>provider_id</otherwise></choose> AS dimension, " +
            "COALESCE(SUM(cost), 0) AS cost FROM sys_ai_billing WHERE cost IS NOT NULL " +
            "<if test='start != null'> AND create_time &gt;= #{start}</if>" +
            "<if test='end != null'> AND create_time &lt;= #{end}</if>" +
            "<if test='modelId != null'> AND model = #{modelId}</if>" +
            "<if test='providerId != null'> AND provider_id = #{providerId}</if>" +
            " GROUP BY dimension ORDER BY dimension</script>")
    List<AiCostStatRead> sumCostGroupBy(@Param("dimension") String dimension,
                                        @Param("start") LocalDateTime start,
                                        @Param("end") LocalDateTime end,
                                        @Param("modelId") String modelId,
                                        @Param("providerId") Long providerId);

    /**
     * 积分流水按来源汇总（账单充值与退款）
     */
    @Select("SELECT source, COALESCE(SUM(amount), 0) AS amount FROM sys_ai_credit_log " +
            "WHERE user_id = #{userId} AND deleted = 0 " +
            "AND create_time >= #{start} AND create_time <= #{end} GROUP BY source")
    List<AiCreditSourceRead> sumCreditLogBySource(@Param("userId") Long userId,
                                                  @Param("start") LocalDateTime start,
                                                  @Param("end") LocalDateTime end);

    /**
     * 指定时间点前最近一笔流水的变动后余额（账期期初/期末余额）
     */
    @Select("SELECT balance_after FROM sys_ai_credit_log WHERE user_id = #{userId} AND deleted = 0 " +
            "AND create_time <= #{end} ORDER BY create_time DESC, id DESC LIMIT 1")
    BigDecimal getBalanceAtOrBefore(@Param("userId") Long userId, @Param("end") LocalDateTime end);

    /**
     * 已实收订单（已支付/已完成）按商品类型汇总，金额单位为分
     */
    @Select("<script>SELECT package_type AS packageType, COALESCE(SUM(paid_amount), 0) AS amount " +
            "FROM sys_order WHERE status IN (2, 3) AND deleted = 0 " +
            "<if test='start != null'> AND paid_time &gt;= #{start}</if>" +
            "<if test='end != null'> AND paid_time &lt;= #{end}</if>" +
            " GROUP BY package_type</script>")
    List<AiOrderIncomeRead> sumPaidOrderByPackageType(@Param("start") LocalDateTime start,
                                                      @Param("end") LocalDateTime end);
}
