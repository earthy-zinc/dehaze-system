/**
 * 消息队列模块（当前未启用）
 *
 * <p><strong>重要说明：</strong>本包下所有类均标注 {@code @ConditionalOnProperty(... havingValue = "true")}，
 * 在当前环境中 <b>完全不加载</b>（{@code rabbitmq.enabled: false}、{@code kafka.enabled: false}）。
 *
 * <p>当前系统所有异步任务均通过 {@code @Async} + {@code datasetTaskExecutor} 线程池执行，
 * 由 {@code TaskExecutorImpl} 中的 {@code ObjectProvider<RabbitMQPublisher>} 判断：
 * <ul>
 *   <li>MQ 已启用 - 发布到 RabbitMQ，由 Consumer 异步消费</li>
 *   <li>MQ 未启用（当前） - fallback 到线程池同步/异步执行</li>
 * </ul>
 *
 * <p>启用 MQ 前需完成：
 * <ol>
 *   <li>配置 {@code rabbitmq.enabled: true} 及连接参数</li>
 *   <li>实现 Consumer 中的 TODO 业务逻辑</li>
 *   <li>配置死信队列（DLX）策略</li>
 * </ol>
 *
 * @see com.pei.dehaze.service.impl.TaskExecutorImpl
 */
package com.pei.dehaze.mq;
