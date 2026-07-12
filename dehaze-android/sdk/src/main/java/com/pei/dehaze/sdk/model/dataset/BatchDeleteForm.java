package com.pei.dehaze.sdk.model.dataset;

import lombok.Data;

import java.util.List;

/**
 * 批量删除表单（对齐后端 BatchDeleteForm）
 */
@Data
public class BatchDeleteForm {
    private List<Long> ids;
    private Boolean force;
}
