package com.pei.clarity.sdk.api;

import com.pei.clarity.sdk.ApiCallback;
import com.pei.clarity.sdk.DehazeSDK;
import com.pei.clarity.sdk.model.Result;
import com.pei.clarity.sdk.model.ai_model.AiModelVO;

import java.util.List;

import retrofit2.Call;

/**
 * AI 模型目录API接口封装
 */
public class AiModelAPI {

    private AiModelAPI() {
    }

    /**
     * 启用模型列表（按登录用户 VIP 等级过滤）
     *
     * @param modelType 模型类型筛选（如图像模型传 image_edit），可为空
     * @param taskTags  任务标签筛选（逗号分隔，含任一即命中），可为空
     */
    public static void listEnabledModels(String modelType, String taskTags,
                                         ApiCallback<List<AiModelVO>> callback) {
        Call<Result<List<AiModelVO>>> call = DehazeSDK.getInstance().getAiModelApiService()
                .listEnabled(modelType, taskTags);
        call.enqueue(callback);
    }
}
