package com.medical.rag.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.reactive.function.client.WebClient;

import java.time.Duration;

@Configuration
public class WebClientConfig {

    @Value("${python.ml-service.url}")
    private String pythonServiceUrl;

    @Bean
    public WebClient pythonWebClient() {
        return WebClient.builder()
                .baseUrl(pythonServiceUrl)
                .build();
    }
}
