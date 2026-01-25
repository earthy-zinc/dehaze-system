package com.pei.dehaze.listener;

import com.pei.dehaze.model.event.ItemFileCreatedEvent;
import com.pei.dehaze.model.event.ItemFileDeletedEvent;
import com.pei.dehaze.service.SysDatasetItemService;
import com.pei.dehaze.service.SysDatasetService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
@DisplayName("DatasetStatsEventListener 单元测试")
class DatasetStatsEventListenerTest {

    @Mock
    private SysDatasetItemService datasetItemService;

    @Mock
    private SysDatasetService datasetService;

    @InjectMocks
    private DatasetStatsEventListener listener;

    @Test
    @DisplayName("处理文件创建事件 - 成功清除缓存")
    void onItemFileCreated_shouldEvictCache() {
        Long itemId = 1L;
        Long fileId = 100L;
        Long datasetId = 10L;

        when(datasetItemService.getDatasetIdByItemId(itemId)).thenReturn(datasetId);

        listener.onItemFileCreated(new ItemFileCreatedEvent(itemId, fileId));

        verify(datasetItemService).getDatasetIdByItemId(itemId);
        verify(datasetService).evictDatasetAndAncestorStatsCache(datasetId);
    }

    @Test
    @DisplayName("处理文件删除事件 - 成功清除缓存")
    void onItemFileDeleted_shouldEvictCache() {
        Long itemId = 1L;
        Long fileId = 100L;
        Long datasetId = 10L;

        when(datasetItemService.getDatasetIdByItemId(itemId)).thenReturn(datasetId);

        listener.onItemFileDeleted(new ItemFileDeletedEvent(itemId, fileId));

        verify(datasetItemService).getDatasetIdByItemId(itemId);
        verify(datasetService).evictDatasetAndAncestorStatsCache(datasetId);
    }

    @Test
    @DisplayName("处理事件 - 数据项不存在时不抛出异常")
    void onItemFileCreated_whenItemNotFound_shouldNotThrow() {
        Long itemId = 999L;
        Long fileId = 100L;

        when(datasetItemService.getDatasetIdByItemId(itemId)).thenReturn(null);

        listener.onItemFileCreated(new ItemFileCreatedEvent(itemId, fileId));

        verify(datasetItemService).getDatasetIdByItemId(itemId);
        verify(datasetService, never()).evictDatasetAndAncestorStatsCache(any());
    }

    @Test
    @DisplayName("处理事件 - 发生异常时不抛出")
    void onItemFileCreated_whenExceptionOccurs_shouldNotThrow() {
        Long itemId = 1L;
        Long fileId = 100L;

        when(datasetItemService.getDatasetIdByItemId(itemId)).thenThrow(new RuntimeException("Database error"));

        listener.onItemFileCreated(new ItemFileCreatedEvent(itemId, fileId));

        verify(datasetItemService).getDatasetIdByItemId(itemId);
        verify(datasetService, never()).evictDatasetAndAncestorStatsCache(any());
    }
}
