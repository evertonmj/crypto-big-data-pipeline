
package com.example.weather.kafka;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Service;

@Service
public class KafkaProducer {

    private static final Logger logger = LoggerFactory.getLogger(KafkaProducer.class);

    private final KafkaTemplate<String, String> kafkaTemplate;

    public KafkaProducer(KafkaTemplate<String, String> kafkaTemplate) {
        this.kafkaTemplate = kafkaTemplate;
    }

    public void sendMessage(String topic, String message) {
        logger.info("Producing message to topic {}: {}", topic, message);
        kafkaTemplate.send(topic, message)
            .addCallback(
                success -> logger.info("Message [{}] delivered with offset {}", message, success.getRecordMetadata().offset()),
                failure -> logger.error("Unable to deliver message [{}]. Cause: {}", message, failure.getMessage())
            );
    }
}
