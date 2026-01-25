package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.*;

import java.util.List;


@Data
@NoArgsConstructor
@AllArgsConstructor
@Schema(name = "BatchUploadResultVO", description = "批量上传结果")
public class BatchUploadResultVO {

    @Schema(description = "总文件数")
    private Integer total;

    @Schema(description = "成功数量")
    private Integer succeeded;

    @Schema(description = "失败数量")
    private Integer failed;

    @Schema(description = "成功项详情列表")
    private List<BatchUploadSuccessItemVO> successItems;

    @Schema(description = "失败项详情列表")
    private List<BatchUploadFailedItemVO> failedItems;
}
