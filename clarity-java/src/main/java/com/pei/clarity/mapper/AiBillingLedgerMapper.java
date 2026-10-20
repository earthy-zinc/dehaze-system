package com.pei.clarity.mapper;

import com.pei.clarity.model.read.AiUserCreditsRead;
import java.math.BigDecimal;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;

/**
 * 余额账本持久化访问（sys_user.credits_balance 相对值 UPDATE）。
 *
 * <p>与 python {@code user_repository} 同口径：Redis 为准实时权威、MySQL 为持久化权威，
 * 落地采用相对值 UPDATE（{@code credits_balance ± amount}）——相对值表达式由单行原子求值，
 * 无需 version 乐观锁（附加 {@code WHERE credits_version=?} 只会让并发更新彼此作废）。
 * Java 侧（管理员调整/退款回补）与 python 推理扣减共用同一账本。
 */
@Mapper
public interface AiBillingLedgerMapper {

    @Select("SELECT credits_balance AS creditsBalance FROM sys_user WHERE id = #{userId} AND deleted = 0")
    AiUserCreditsRead getCreditsBalance(@Param("userId") Long userId);

    @Update("UPDATE sys_user SET credits_balance = credits_balance + #{amount} "
            + "WHERE id = #{userId} AND deleted = 0")
    int increaseBalance(@Param("userId") Long userId, @Param("amount") BigDecimal amount);

    @Update("UPDATE sys_user SET credits_balance = credits_balance - #{amount} "
            + "WHERE id = #{userId} AND deleted = 0")
    int deductBalance(@Param("userId") Long userId, @Param("amount") BigDecimal amount);
}
