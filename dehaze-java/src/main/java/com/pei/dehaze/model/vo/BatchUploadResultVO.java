package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.*;

import java.util.List;


@Data
@NoArgsConstructor
@AllArgsConstructor
@EqualsAndHashCode(callSuper = true)
@Schema(name = "BatchUploadResultVO", description = "批量上传结果")
public class BatchUploadResultVO extends BatchOperationResultVO {

    @Schema(description = "总文件数")
    private Integer totalFiles;

    @Schema(description = "成功配对详情")
    private List<DatasetItemVO> successDetails;
}
