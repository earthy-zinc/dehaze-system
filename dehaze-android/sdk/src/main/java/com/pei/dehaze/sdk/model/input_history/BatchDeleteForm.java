package com.pei.dehaze.sdk.model.input_history;

import java.util.List;

import lombok.Data;

/**
 * 批量删除表单
 */
@Data
public class BatchDeleteForm {
    private List<Long> ids;
}
