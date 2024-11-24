
package com.example.weather.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

@Service
public class TickersMarketService {

    @Value("${blockchain.api.exchange}")
    private String blockchainApi;

    @Value("${blockchain.api.trades-pair}")
    private String filterPair;

    private final RestTemplate restTemplate;
    private final ObjectMapper objectMapper;

    public TickersMarketService(RestTemplate restTemplate, ObjectMapper objectMapper) {
        this.restTemplate = restTemplate;
        this.objectMapper = objectMapper;
    }

    public JsonNode fetchTradesFromPair(String symbol) throws Exception {
        String tickersUrl = blockchainApi + filterPair + symbol;
        String response = restTemplate.getForObject(tickersUrl, String.class);
        return objectMapper.readTree(response).get("data");
    }

}
