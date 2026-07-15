package com.pei.dehaze.service.strategy;

import com.pei.dehaze.common.exception.BusinessException;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * 任务策略工厂
 * 根据任务类型获取对应的策略实现
 */
@Slf4j
@Component
public class TaskStrategyFactory {

    private final Map<String, TaskStrategy> strategyMap;

    public TaskStrategyFactory(List<TaskStrategy> strategies) {
        this.strategyMap = strategies.stream()
                .collect(Collectors.toMap(
                        TaskStrategy::getTaskType,
                        Function.identity(),
                        (existing, replacement) -> {
                            log.warn("Duplicate strategy for task type: {}, replacing", existing.getTaskType());
                            return replacement;
                        }
                ));
        log.info("Initialized TaskStrategyFactory with {} strategies: {}",
                strategyMap.size(), strategyMap.keySet());
    }

    /**
     * 根据任务类型获取对应的策略实现
     * @param taskType 任务类型
     * @return 策略实现
     * @throws BusinessException 当找不到对应策略时抛出
     */
    public TaskStrategy getStrategy(String taskType) {
        TaskStrategy strategy = strategyMap.get(taskType);
        if (strategy == null) {
            throw new BusinessException("Unsupported task type: " + taskType);
        }
        return strategy;
    }
}
