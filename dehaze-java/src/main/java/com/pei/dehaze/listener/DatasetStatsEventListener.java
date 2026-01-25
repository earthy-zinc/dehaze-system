package com.pei.dehaze.listener;

import com.pei.dehaze.model.event.ItemFileCreatedEvent;
import com.pei.dehaze.model.event.ItemFileDeletedEvent;
import com.pei.dehaze.service.SysDatasetItemService;
import com.pei.dehaze.service.SysDatasetService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.context.event.EventListener;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Component;
import org.springframework.transaction.event.TransactionPhase;
import org.springframework.transaction.event.TransactionalEventListener;

/**
 * 数据集统计事件监听器
 * 监听文件创建/删除事件，异步更新数据集统计缓存
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class DatasetStatsEventListener {

    private final SysDatasetItemService datasetItemService;
    private final SysDatasetService datasetService;

    /**
     * 处理文件创建事件 - 在事务提交后异步执行
     */
    @TransactionalEventListener(phase = TransactionPhase.AFTER_COMMIT)
    @Async
    public void onItemFileCreated(ItemFileCreatedEvent event) {
        log.debug("处理文件创建事件: itemId={}, fileId={}", event.itemId(), event.fileId());
        evictDatasetStatsCache(event.itemId());
    }

    /**
     * 处理文件删除事件 - 在事务提交后异步执行
     */
    @TransactionalEventListener(phase = TransactionPhase.AFTER_COMMIT)
    @Async
    public void onItemFileDeleted(ItemFileDeletedEvent event) {
        log.debug("处理文件删除事件: itemId={}, fileId={}", event.itemId(), event.fileId());
        evictDatasetStatsCache(event.itemId());
    }

    /**
     * 清除数据集及其祖先的统计缓存
     */
    private void evictDatasetStatsCache(Long itemId) {
        try {
            Long datasetId = datasetItemService.getDatasetIdByItemId(itemId);
            if (datasetId != null) {
                datasetService.evictDatasetAndAncestorStatsCache(datasetId);
                log.debug("已清除数据集统计缓存: datasetId={}", datasetId);
            }
        } catch (Exception e) {
            log.warn("清除数据集统计缓存失败: itemId={}, error={}", itemId, e.getMessage());
        }
    }
}
