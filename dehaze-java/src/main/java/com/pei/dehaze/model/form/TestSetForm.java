package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.Size;
import lombok.Data;

import java.util.List;

@Schema(description = "召回测试集创建表单")
@Data
public class TestSetForm {

    @NotBlank(message = "测试问题不能为空")
    @Size(min = 1, max = 1000)
    private String question;

    @NotEmpty(message = "期望命中分块不能为空")
    private List<Long> expectedChunkIds;
}
