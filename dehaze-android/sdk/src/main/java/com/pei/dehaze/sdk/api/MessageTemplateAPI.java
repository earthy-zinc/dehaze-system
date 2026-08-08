package com.pei.dehaze.sdk.api;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.DehazeSDK;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.message.MessageTemplateForm;
import com.pei.dehaze.sdk.model.message.MessageTemplateQuery;
import com.pei.dehaze.sdk.model.message.MessageTemplateVO;

import retrofit2.Call;

/**
 * 消息模板管理API接口封装
 * 对齐后端路由：/api/v1/message-templates
 */
public class MessageTemplateAPI {

    private MessageTemplateAPI() {
    }

    public static void getPage(MessageTemplateQuery query, ApiCallback<PageResult<MessageTemplateVO>> callback) {
        Call<Result<PageResult<MessageTemplateVO>>> call = DehazeSDK.getInstance().getMessageTemplateApiService().getPage(
                query.getPageNum(), query.getPageSize(),
                query.getName(), query.getType(), query.getStatus());
        call.enqueue(callback);
    }

    public static void getDetail(long id, ApiCallback<MessageTemplateVO> callback) {
        Call<Result<MessageTemplateVO>> call = DehazeSDK.getInstance().getMessageTemplateApiService().getDetail(id);
        call.enqueue(callback);
    }

    public static void update(long id, MessageTemplateForm form, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getMessageTemplateApiService().update(id, form);
        call.enqueue(callback);
    }
}
