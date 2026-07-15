package com.pei.dehaze.listener;

import com.pei.dehaze.model.event.ItemFileCreatedEvent;
import com.pei.dehaze.model.event.ItemFileDeletedEvent;
import com.pei.dehaze.service.SysDatasetService;
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
    private SysDatasetService datasetService;

    @InjectMocks
    private DatasetStatsEventListener listener;

    @Test
    @DisplayName("处理文件创建事件 - 成功清除缓存")
    void onItemFileCreated_shouldEvictCache() {
        Long itemId = 1L;
        Long fileId = 100L;

        listener.onItemFileCreated(new ItemFileCreatedEvent(itemId, fileId));

        verify(datasetService).evictAllDatasetsCache();
    }

    @Test
    @DisplayName("处理文件删除事件 - 成功清除缓存")
    void onItemFileDeleted_shouldEvictCache() {
        Long itemId = 1L;
        Long fileId = 100L;

        listener.onItemFileDeleted(new ItemFileDeletedEvent(itemId, fileId));

        verify(datasetService).evictAllDatasetsCache();
    }

    @Test
    @DisplayName("处理事件 - 清除缓存异常时不抛出")
    void onItemFileCreated_whenEvictionFails_shouldNotThrow() {
        Long itemId = 1L;
        Long fileId = 100L;

        doThrow(new RuntimeException("Cache error")).when(datasetService).evictAllDatasetsCache();

        listener.onItemFileCreated(new ItemFileCreatedEvent(itemId, fileId));

        verify(datasetService).evictAllDatasetsCache();
    }
}
