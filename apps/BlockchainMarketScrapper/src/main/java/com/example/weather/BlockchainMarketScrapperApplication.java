
package com.example.weather;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableScheduling;

@SpringBootApplication
@EnableScheduling
public class BlockchainMarketScrapperApplication {
    public static void main(String[] args) {
        SpringApplication.run(BlockchainMarketScrapperApplication.class, args);
    }
}
