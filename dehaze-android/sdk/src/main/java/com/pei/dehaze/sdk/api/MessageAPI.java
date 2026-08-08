package com.pei.dehaze.sdk.api;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.DehazeSDK;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.message.MessageQuery;
import com.pei.dehaze.sdk.model.message.MessageSearchQuery;
import com.pei.dehaze.sdk.model.message.MessageSendRequest;
import com.pei.dehaze.sdk.model.message.MessageSendResult;
import com.pei.dehaze.sdk.model.message.MessageVO;
import com.pei.dehaze.sdk.model.message.ReadAllResult;
import com.pei.dehaze.sdk.model.message.UnreadCountVO;

import retrofit2.Call;

/**
 * 消息管理API接口封装
 * 对齐后端路由：/api/v1/messages
 */
public class MessageAPI {

    private MessageAPI() {
    }

    public static void getPage(MessageQuery query, ApiCallback<PageResult<MessageVO>> callback) {
        Call<Result<PageResult<MessageVO>>> call = DehazeSDK.getInstance().getMessageApiService().getPage(
                query.getPageNum(), query.getPageSize(), query.getType(), query.getReadStatus());
        call.enqueue(callback);
    }

    public static void getUnreadCount(ApiCallback<UnreadCountVO> callback) {
        Call<Result<UnreadCountVO>> call = DehazeSDK.getInstance().getMessageApiService().getUnreadCount();
        call.enqueue(callback);
    }

    public static void getDetail(long id, ApiCallback<MessageVO> callback) {
        Call<Result<MessageVO>> call = DehazeSDK.getInstance().getMessageApiService().getDetail(id);
        call.enqueue(callback);
    }

    public static void markRead(long id, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getMessageApiService().markRead(id);
        call.enqueue(callback);
    }

    public static void markAllRead(String type, ApiCallback<ReadAllResult> callback) {
        Call<Result<ReadAllResult>> call = DehazeSDK.getInstance().getMessageApiService().markAllRead(type);
        call.enqueue(callback);
    }

    public static void deleteByIds(String ids, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getMessageApiService().deleteByIds(ids);
        call.enqueue(callback);
    }

    public static void search(MessageSearchQuery query, ApiCallback<PageResult<MessageVO>> callback) {
        Call<Result<PageResult<MessageVO>>> call = DehazeSDK.getInstance().getMessageApiService().search(
                query.getPageNum(), query.getPageSize(), query.getKeyword());
        call.enqueue(callback);
    }

    public static void send(MessageSendRequest request, ApiCallback<MessageSendResult> callback) {
        Call<Result<MessageSendResult>> call = DehazeSDK.getInstance().getMessageApiService().send(request);
        call.enqueue(callback);
    }
}
