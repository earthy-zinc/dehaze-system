package com.pei.dehaze.model.read;

import lombok.Data;

/** 账单按计费类型汇总行 */
@Data
public class AiBillTypeRead {

    private String billType;

    private Long credits;
}
