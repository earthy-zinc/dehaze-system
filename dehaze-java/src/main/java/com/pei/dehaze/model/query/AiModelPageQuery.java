package com.pei.dehaze.model.query;

import com.pei.dehaze.common.base.BasePageQuery;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Schema(description = "AI 模型分页查询参数")
@Data
@EqualsAndHashCode(callSuper = true)
public class AiModelPageQuery extends BasePageQuery {

    @Schema(description = "关键字(按显示名称/模型标识模糊搜索)")
    private String keyword;

    @Schema(description = "按模型类型筛选(chat/embedding/rerank/image_gen/image_edit/video_gen/asr/tts/music_gen/3d_gen)")
    private String modelType;

    @Schema(description = "按任务标签筛选(逗号分隔，task_tags 数组含任一即命中)")
    private String taskTags;
}
