package com.pei.dehaze.service.impl;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.Base64;

/**
 * 知识库分块与记忆向量的 ES 清理原语。
 *
 * <p>索引生命周期（建/删/统计，{@code POST /kb}、{@code DELETE /kb/{id}}、
 * {@code GET /kb/{id}/index-stats}）归 java-proxy；本类承载文档更新/删除时须同步清除的分块向量
 * （索引名 {@code kb_chunks_{kbId}}）与记忆删除时的向量文档清理（索引名 {@code ai_memory}），
 * 避免 MySQL 已清而 ES 残留孤儿文档。
 */
@Slf4j
@Component
public class AiKbIndexClient {

    private static final String INDEX_PREFIX = "kb_chunks_";

    /** 记忆向量索引（与 python `ai_memory_index.INDEX_NAME` 同名） */
    private static final String MEMORY_INDEX = "ai_memory";
    private static final Duration TIMEOUT = Duration.ofSeconds(5);

    private final String esUrl;
    private final String authHeader;
    private final HttpClient httpClient = HttpClient.newBuilder()
            .connectTimeout(TIMEOUT)
            .build();

    public AiKbIndexClient(@Value("${ES_URL:}") String esUrl,
                           @Value("${ES_USERNAME:}") String username,
                           @Value("${ES_PASSWORD:}") String password) {
        this.esUrl = esUrl == null ? "" : esUrl.replaceAll("/+$", "");
        this.authHeader = username == null || username.isBlank()
                ? null
                : "Basic " + Base64.getEncoder().encodeToString(
                        (username + ":" + password).getBytes(StandardCharsets.UTF_8));
    }

    /** 清除文档在 ES 中的全部分块（文档删除/版本更新时调用），失败不阻断主流程 */
    public boolean deleteDocumentChunks(Long kbId, Long documentId) {
        if (esUrl.isBlank()) {
            return false;
        }
        String index = INDEX_PREFIX + kbId;
        try {
            String body = "{\"query\":{\"term\":{\"doc_id\":" + documentId + "}}}";
            HttpResponse<String> response = httpClient.send(
                    builder(esUrl + "/" + index + "/_delete_by_query?refresh=true")
                            .POST(HttpRequest.BodyPublishers.ofString(body, StandardCharsets.UTF_8))
                            .build(),
                    HttpResponse.BodyHandlers.ofString());
            // 索引不存在（404）视为已清理
            return response.statusCode() < 300 || response.statusCode() == 404;
        } catch (Exception e) {
            log.warn("ES 文档分块清理失败 index={} docId={}: {}", index, documentId, e.getMessage());
            return false;
        }
    }

    /** 清除记忆的 ES 向量文档（文档 _id 为记忆 id）；已删记忆不清理向量会被检索召回，失败只告警不阻断 */
    public boolean deleteMemoryDoc(Long memoryId) {
        if (esUrl.isBlank()) {
            return false;
        }
        try {
            HttpResponse<String> response = httpClient.send(
                    builder(esUrl + "/" + MEMORY_INDEX + "/_doc/" + memoryId + "?refresh=true")
                            .DELETE().build(),
                    HttpResponse.BodyHandlers.ofString());
            // 文档不存在（404）视为已清理
            return response.statusCode() < 300 || response.statusCode() == 404;
        } catch (Exception e) {
            log.warn("ES 记忆向量文档清理失败 memoryId={}: {}", memoryId, e.getMessage());
            return false;
        }
    }

    private HttpRequest.Builder builder(String url) {
        HttpRequest.Builder builder = HttpRequest.newBuilder(URI.create(url)).timeout(TIMEOUT);
        if (authHeader != null) {
            builder.header("Authorization", authHeader);
        }
        return builder.header("Content-Type", "application/json");
    }
}
