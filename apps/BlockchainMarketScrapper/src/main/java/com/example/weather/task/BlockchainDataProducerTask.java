
package com.example.weather.task;

import com.example.weather.kafka.KafkaProducer;
import com.example.weather.service.TickersMarketService;
import com.fasterxml.jackson.databind.JsonNode;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.CommandLineRunner;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.function.Function;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.Arrays;
import java.util.HashMap;

@Component
public class BlockchainDataProducerTask implements CommandLineRunner {

    private final TickersMarketService tickersMarketService;
    private final KafkaProducer kafkaProducer;
    private List<String> pairsList;
    private Logger logger = Logger.getLogger(BlockchainDataProducerTask.class.getName());

    @Value("${kafka.topic.name}")
    private String kafkaTopic;

    @Value("${blockchain.pairs}")
    private String pairs;

    public BlockchainDataProducerTask(TickersMarketService tickersMarketService, KafkaProducer kafkaProducer) {
        this.tickersMarketService = tickersMarketService;
        this.kafkaProducer = kafkaProducer;
    }

    @Override
    public void run(String... args) throws Exception {
        // Start producing data on application startup
        fetchAndProduceData();
    }

    @Scheduled(fixedRateString = "${fetch.interval.ms}")
    public void fetchAndProduceData() {
        try {
            List<String> pairsList = new ArrayList<>(Arrays.asList(pairs.split(",")));

            for (String pair : pairsList) {
                logger.info("Fetching data for pair: " + pair);
                JsonNode pairTrades = (tickersMarketService.fetchTradesFromPair(pair));
                pairTrades.forEach(this::produceData);
            }
        } catch (Exception e) {
            logger.severe("Error fetching data from the API: " + e.getMessage());
            e.printStackTrace();
        }

    }

    private void produceData(JsonNode tickerData) {
        logger.info("Producing message: " + tickerData);
        String message = convertTickerDataToMessage(tickerData);
        
        kafkaProducer.sendMessage(kafkaTopic, message);
    }

    private String convertTickerDataToMessage(JsonNode tickerData) {
        Long id = tickerData.get("id").asLong();
        Double amount = tickerData.get("amount").asDouble();
        BigDecimal rate = BigDecimal.valueOf(tickerData.get("rate").asLong());
        String pair = tickerData.get("pair").asText();
        String orderType = tickerData.get("order_type").asText();
        String timestamp = tickerData.get("created_at").asText();

        return String.format("{\"id\": %d, \"amount\": %.2f, \"rate\": %.2f, \"pair\": \"%s\", \"order_type\": \"%s\", \"created_at\": \"%s\"}",
                id, amount, rate, pair, orderType, timestamp);    }
}
