package com.pei.dehaze.model.read;

import lombok.Data;

import java.math.BigDecimal;

/** 用户积分余额与乐观锁版本号（余额 CAS 前置读取） */
@Data
public class AiUserCreditsRead {

    private BigDecimal creditsBalance;

    private Integer creditsVersion;
}
