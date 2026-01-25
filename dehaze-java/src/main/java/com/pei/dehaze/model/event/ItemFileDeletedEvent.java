package com.pei.dehaze.model.event;

/**
 * 数据项文件删除事件
 * 用于解耦文件服务与数据集服务之间的循环依赖
 *
 * @param itemId 数据项ID
 * @param fileId 文件ID
 */
public record ItemFileDeletedEvent(Long itemId, Long fileId) {
}
