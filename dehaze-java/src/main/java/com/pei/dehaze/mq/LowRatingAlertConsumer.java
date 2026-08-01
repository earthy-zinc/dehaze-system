package com.pei.dehaze.mq;

import com.pei.dehaze.model.entity.SysRating;
import com.pei.dehaze.service.LowRatingAlertService;
import com.pei.dehaze.service.RatingService;
import com.rabbitmq.client.Channel;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.amqp.core.Message;
import org.springframework.amqp.rabbit.annotation.RabbitListener;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

@Slf4j
@Component
@RequiredArgsConstructor
@ConditionalOnProperty(prefix = "rabbitmq", name = "enabled", havingValue = "true")
public class LowRatingAlertConsumer extends RabbitMQConsumer {

    private static final String QUEUE_LOW_RATING_ALERT = "feedback.low_rating";

    private final LowRatingAlertService lowRatingAlertService;
    private final RatingService ratingService;

    @RabbitListener(queues = QUEUE_LOW_RATING_ALERT)
    public void onLowRatingAlert(Message message, Channel channel) {
        processMessage(message, channel, QUEUE_LOW_RATING_ALERT, this::handleLowRatingAlert);
    }

    private void handleLowRatingAlert(String body, String traceId) throws Exception {
        Long ratingId = Long.parseLong(body.trim());
        SysRating rating = ratingService.getById(ratingId);
        if (rating == null) {
            log.warn("低分告警：评价不存在，跳过。ratingId={}", ratingId);
            return;
        }
        if (rating.getRating() == null || rating.getRating() > 2) {
            log.debug("低分告警：评分高于2，跳过。ratingId={}, rating={}", ratingId, rating.getRating());
            return;
        }
        lowRatingAlertService.checkAndAlert(rating);
        log.debug("低分告警处理完成: ratingId={}, traceId={}", ratingId, traceId);
    }
}
