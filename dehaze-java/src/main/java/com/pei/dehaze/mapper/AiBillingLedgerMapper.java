package com.pei.dehaze.mapper;

import com.pei.dehaze.model.read.AiUserCreditsRead;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;

import java.math.BigDecimal;

/**
 * 余额账本持久化访问（sys_user.credits_balance + credits_version 乐观锁 CAS）。
 *
 * <p>与 python {@code user_repository} 同口径：Redis 为准实时权威、MySQL 为持久化权威，
 * 落地采用 version 字段 CAS，Java 侧（管理员调整/退款回补）与 python 推理扣减共用同一账本。
 */
@Mapper
public interface AiBillingLedgerMapper {

    @Select("SELECT credits_balance AS creditsBalance, credits_version AS creditsVersion " +
            "FROM sys_user WHERE id = #{userId} AND deleted = 0")
    AiUserCreditsRead getCreditsBalanceAndVersion(@Param("userId") Long userId);

    @Update("UPDATE sys_user SET credits_balance = credits_balance + #{amount}, " +
            "credits_version = credits_version + 1 WHERE id = #{userId} AND credits_version = #{version}")
    int increaseBalanceCas(@Param("userId") Long userId,
                           @Param("amount") BigDecimal amount,
                           @Param("version") Integer version);

    @Update("UPDATE sys_user SET credits_balance = credits_balance - #{amount}, " +
            "credits_version = credits_version + 1 WHERE id = #{userId} AND credits_version = #{version}")
    int deductBalanceCas(@Param("userId") Long userId,
                         @Param("amount") BigDecimal amount,
                         @Param("version") Integer version);
}
