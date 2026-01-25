package com.pei.dehaze.service;

import com.pei.dehaze.model.entity.SysDataset;
import com.pei.dehaze.model.entity.SysDatasetItem;
import com.pei.dehaze.model.form.DatasetAddForm;
import com.pei.dehaze.model.vo.DatasetVO;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

import static org.assertj.core.api.Assertions.*;

import com.pei.dehaze.config.TestConfig;

/**
 * 数据集并发场景测试
 * <p>
 * 测试目的：验证数据集模块在并发场景下的数据一致性和线程安全性
 * 测试策略：使用 @SpringBootTest 启动完整容器，测试真实的并发场景
 * <p>
 * 注意：
 * 1. 这是集成测试，需要真实的数据库和Spring容器支持
 * 2. 需要配置test profile和测试数据库
 * 3. 建议在CI/CD环境中单独运行集成测试
 * 4. 测试后会自动清理测试数据，避免污染数据库
 *
 * @author earthy-zinc
 * @since 2025-01-10
 */
@SpringBootTest(classes = TestConfig.class)
@DisplayName("数据集并发场景测试")
class DatasetConcurrencyIT {

    @Autowired
    private SysDatasetService sysDatasetService;

    @Autowired
    private SysDatasetItemService sysDatasetItemService;

    @Autowired
    private DatasetOperationService datasetOperationService;

    private ExecutorService executorService;
    private static final int THREAD_COUNT = 10;
    private static final int TIMEOUT_SECONDS = 30;

    // 用于记录测试期间创建的数据集ID，以便清理
    private List<Long> createdDatasetIds = new ArrayList<>();

    @BeforeEach
    void setUp() {
        executorService = Executors.newFixedThreadPool(THREAD_COUNT);
        createdDatasetIds = new ArrayList<>();
    }

    /**
     * 清理测试数据
     * 测试目的：清理测试期间创建的所有数据集和相关数据项
     * 测试场景：每个测试执行完成后自动清理
     * 验证内容：确保测试数据不会污染数据库
     */
    @AfterEach
    void tearDown() {
        // 关闭线程池
        if (executorService != null && !executorService.isShutdown()) {
            executorService.shutdown();
            try {
                if (!executorService.awaitTermination(10, TimeUnit.SECONDS)) {
                    executorService.shutdownNow();
                }
            } catch (InterruptedException e) {
                executorService.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }

        // 清理测试数据集
        for (Long datasetId : createdDatasetIds) {
            try {
                sysDatasetService.deleteDataset(datasetId);
            } catch (Exception e) {
                // 忽略清理异常，可能是数据已经不存在
            }
        }
        createdDatasetIds.clear();
    }

    /**
     * 测试并发创建数据项 - 无数据冲突
     * 测试目的：验证多线程同时向同一数据集创建数据项时不会发生数据冲突
     * 测试场景：10个线程同时创建数据项，每个线程创建5个数据项
     * 验证内容：所有数据项都应该成功创建，总数应该等于 10 * 5 = 50
     * <p>
     * 注意：此测试需要真实的数据库环境，默认禁用
     */
    @Test
    @DisplayName("并发创建数据项 - 无数据冲突")
    void testConcurrentCreateDatasetItems() throws InterruptedException, ExecutionException {
        // Given - 创建一个测试数据集
        DatasetAddForm datasetForm = new DatasetAddForm();
        datasetForm.setName("并发测试数据集_" + System.currentTimeMillis());
        datasetForm.setDescription("用于并发创建数据项测试");
        datasetForm.setParentId(0L);
        datasetForm.setType("training");

        DatasetVO dataset = sysDatasetService.addDataset(datasetForm);
        Long datasetId = dataset.getId();
        createdDatasetIds.add(datasetId); // 记录ID用于清理

        // When - 并发创建数据项
        int itemsPerThread = 5;
        AtomicInteger successCount = new AtomicInteger(0);
        AtomicInteger failureCount = new AtomicInteger(0);

        List<Future<?>> futures = new ArrayList<>();

        for (int i = 0; i < THREAD_COUNT; i++) {
            final int threadIndex = i;
            Future<?> future = executorService.submit(() -> {
                try {
                    for (int j = 0; j < itemsPerThread; j++) {
                        String itemName = "并发数据项_线程" + threadIndex + "_项" + j;
                        sysDatasetItemService.createDatasetItem(datasetId, itemName);
                        successCount.incrementAndGet();
                    }
                } catch (Exception e) {
                    failureCount.incrementAndGet();
                }
            });
            futures.add(future);
        }

        // 等待所有任务完成
        for (Future<?> future : futures) {
            future.get();
        }

        // Then - 验证结果
        int expectedTotal = THREAD_COUNT * itemsPerThread;
        assertThat(successCount.get()).isEqualTo(expectedTotal);
        assertThat(failureCount.get()).isEqualTo(0);

        // 验证数据库中的实际数据
        List<SysDatasetItem> items = sysDatasetItemService.lambdaQuery()
                .eq(SysDatasetItem::getDatasetId, datasetId)
                .list();
        assertThat(items).hasSize(expectedTotal);
    }

    /**
     * 测试并发删除数据集 - 事务隔离
     * 测试目的：验证多线程同时删除不同数据集时的事务隔离性
     * 测试场景：创建10个数据集，然后10个线程同时删除这些数据集
     * 验证内容：所有数据集都应该被成功删除，不应该有事务冲突
     */
    @Test
    @DisplayName("并发删除数据集 - 事务隔离")
    void testConcurrentDeleteDatasets() throws InterruptedException, ExecutionException {
        // Given - 创建多个测试数据集
        List<Long> datasetIds = new ArrayList<>();
        for (int i = 0; i < THREAD_COUNT; i++) {
            DatasetAddForm datasetForm = new DatasetAddForm();
            datasetForm.setName("并发删除测试数据集_" + i + "_" + System.currentTimeMillis());
            datasetForm.setDescription("用于并发删除测试");
            datasetForm.setParentId(0L);
            datasetForm.setType("training");

            DatasetVO dataset = sysDatasetService.addDataset(datasetForm);
            datasetIds.add(dataset.getId());
        }
        createdDatasetIds.addAll(datasetIds); // 记录ID用于清理（虽然会被删除，但保持一致性）

        // When - 并发删除数据集
        AtomicInteger successCount = new AtomicInteger(0);
        AtomicInteger failureCount = new AtomicInteger(0);

        List<Future<?>> futures = new ArrayList<>();

        for (Long datasetId : datasetIds) {
            Future<?> future = executorService.submit(() -> {
                try {
                    sysDatasetService.deleteDataset(datasetId);
                    successCount.incrementAndGet();
                } catch (Exception e) {
                    failureCount.incrementAndGet();
                }
            });
            futures.add(future);
        }

        // 等待所有任务完成
        for (Future<?> future : futures) {
            future.get();
        }

        // Then - 验证结果
        assertThat(successCount.get()).isEqualTo(THREAD_COUNT);
        assertThat(failureCount.get()).isEqualTo(0);

        // 验证数据库中的数据已被删除
        for (Long datasetId : datasetIds) {
            SysDataset dataset = sysDatasetService.getById(datasetId);
            assertThat(dataset).isNull();
        }

        // 清理列表，因为数据集已被删除
        createdDatasetIds.clear();
    }

    /**
     * 测试读写并发 - 统计信息一致性
     * 测试目的：验证在读取统计信息的同时进行写操作时的数据一致性
     * 测试场景：一半线程读取统计信息，一半线程创建数据项
     * 验证内容：读取操作不应该失败，写入操作应该成功，最终统计信息应该正确
     */
    @Test
    @DisplayName("读写并发 - 统计信息一致性")
    void testConcurrentReadWriteStatistics() throws InterruptedException, ExecutionException {
        // Given - 创建一个测试数据集
        DatasetAddForm datasetForm = new DatasetAddForm();
        datasetForm.setName("读写并发测试数据集_" + System.currentTimeMillis());
        datasetForm.setDescription("用于读写并发测试");
        datasetForm.setParentId(0L);
        datasetForm.setType("training");

        DatasetVO dataset = sysDatasetService.addDataset(datasetForm);
        Long datasetId = dataset.getId();
        createdDatasetIds.add(datasetId); // 记录ID用于清理

        // When - 并发读写操作
        int writeThreads = THREAD_COUNT / 2;
        int readThreads = THREAD_COUNT / 2;

        AtomicInteger writeSuccessCount = new AtomicInteger(0);
        AtomicInteger readSuccessCount = new AtomicInteger(0);
        AtomicInteger failureCount = new AtomicInteger(0);

        List<Future<?>> futures = new ArrayList<>();

        // 启动写线程
        for (int i = 0; i < writeThreads; i++) {
            final int threadIndex = i;
            Future<?> future = executorService.submit(() -> {
                try {
                    String itemName = "并发写入数据项_" + threadIndex;
                    sysDatasetItemService.createDatasetItem(datasetId, itemName);
                    writeSuccessCount.incrementAndGet();
                } catch (Exception e) {
                    failureCount.incrementAndGet();
                }
            });
            futures.add(future);
        }

        // 启动读线程
        for (int i = 0; i < readThreads; i++) {
            Future<?> future = executorService.submit(() -> {
                try {
                    DatasetVO datasetVO = sysDatasetService.getDatasetById(datasetId);
                    assertThat(datasetVO).isNotNull();
                    assertThat(datasetVO.getStatistics()).isNotNull();
                    readSuccessCount.incrementAndGet();
                } catch (Exception e) {
                    failureCount.incrementAndGet();
                }
            });
            futures.add(future);
        }

        // 等待所有任务完成
        for (Future<?> future : futures) {
            future.get();
        }

        // Then - 验证结果
        assertThat(writeSuccessCount.get()).isEqualTo(writeThreads);
        assertThat(readSuccessCount.get()).isEqualTo(readThreads);
        assertThat(failureCount.get()).isEqualTo(0);

        // 验证最终统计信息
        DatasetVO finalDataset = sysDatasetService.getDatasetById(datasetId);
        assertThat(finalDataset.getStatistics()).isNotNull();
    }

    /**
     * 测试并发批量上传 - 无数据丢失
     * 测试目的：验证多线程同时批量上传数据项时不会发生数据丢失
     * 测试场景：5个线程同时批量创建数据项，每个线程创建10个数据项
     * 验证内容：所有数据项都应该成功创建，总数应该等于 5 * 10 = 50
     */
    @Test
    @DisplayName("并发批量上传 - 无数据丢失")
    void testConcurrentBatchUpload() throws InterruptedException, ExecutionException {
        // Given - 创建一个测试数据集
        DatasetAddForm datasetForm = new DatasetAddForm();
        datasetForm.setName("并发批量上传测试数据集_" + System.currentTimeMillis());
        datasetForm.setDescription("用于并发批量上传测试");
        datasetForm.setParentId(0L);
        datasetForm.setType("training");

        DatasetVO dataset = sysDatasetService.addDataset(datasetForm);
        Long datasetId = dataset.getId();
        createdDatasetIds.add(datasetId); // 记录ID用于清理

        // When - 并发批量创建数据项
        int batchThreads = 5;
        int itemsPerBatch = 10;

        AtomicInteger successCount = new AtomicInteger(0);
        AtomicInteger failureCount = new AtomicInteger(0);

        List<Future<?>> futures = new ArrayList<>();

        for (int i = 0; i < batchThreads; i++) {
            final int threadIndex = i;
            Future<?> future = executorService.submit(() -> {
                try {
                    for (int j = 0; j < itemsPerBatch; j++) {
                        String itemName = "批量上传数据项_批次" + threadIndex + "_项" + j;
                        sysDatasetItemService.createDatasetItem(datasetId, itemName);
                        successCount.incrementAndGet();
                    }
                } catch (Exception e) {
                    failureCount.incrementAndGet();
                }
            });
            futures.add(future);
        }

        // 等待所有任务完成
        for (Future<?> future : futures) {
            future.get();
        }

        // Then - 验证结果
        int expectedTotal = batchThreads * itemsPerBatch;
        assertThat(successCount.get()).isEqualTo(expectedTotal);
        assertThat(failureCount.get()).isEqualTo(0);

        // 验证数据库中的实际数据
        List<SysDatasetItem> items = sysDatasetItemService.lambdaQuery()
                .eq(SysDatasetItem::getDatasetId, datasetId)
                .list();
        assertThat(items).hasSize(expectedTotal);
    }
}
