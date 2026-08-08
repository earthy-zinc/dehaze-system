package com.pei.dehaze.sdk.logger;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

import java.io.IOException;
import java.util.List;

import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

/**
 * 生产环境 transport：批量上报后端接收 API（POST /api/v1/logs/client）。
 */
public class RemoteTransport implements LogTransport {

    private static final MediaType JSON = MediaType.parse("application/json; charset=utf-8");
    private static final String ENDPOINT = "/api/v1/logs/client";
    private static final int MAX_BATCH = 50;

    private final OkHttpClient client;
    private final String baseUrl;
    private final Gson gson = new Gson();

    public RemoteTransport(OkHttpClient client, String baseUrl) {
        this.client = client;
        this.baseUrl = baseUrl;
    }

    @Override
    public void log(LogEntry entry) {
        // 生产环境不在控制台逐条刷屏
    }

    @Override
    public void send(List<LogEntry> logs) throws Exception {
        if (logs == null || logs.isEmpty()) {
            return;
        }
        List<LogEntry> batch = logs.size() > MAX_BATCH
                ? logs.subList(0, MAX_BATCH) : logs;

        JsonArray arr = new JsonArray();
        for (LogEntry entry : batch) {
            arr.add(entry.toJson());
        }
        JsonObject body = new JsonObject();
        body.add("logs", arr);

        Request request = new Request.Builder()
                .url(baseUrl + ENDPOINT)
                .post(RequestBody.create(gson.toJson(body), JSON))
                .build();

        try (Response response = client.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                throw new IOException("remote log upload failed: " + response.code());
            }
        }
    }
}
