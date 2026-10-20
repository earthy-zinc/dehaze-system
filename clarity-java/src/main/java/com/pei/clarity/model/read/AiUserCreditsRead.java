package com.pei.clarity.model.read;

import java.math.BigDecimal;
import lombok.Data;

/** 用户积分余额（余额落库前置读取） */
@Data
public class AiUserCreditsRead {

    private BigDecimal creditsBalance;
}
