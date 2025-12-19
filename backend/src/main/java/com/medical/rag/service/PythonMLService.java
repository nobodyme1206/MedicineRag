package com.medical.rag.service;

import com.medical.rag.dto.AgentRequest;
import com.medical.rag.dto.QuestionRequest;
import com.medical.rag.dto.RetrievalRequest;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.time.Duration;
import java.util.Map;

/**
 * Python ML微服务调用
 * 负责与Python后端通信，调用RAG、Agent等ML功能
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class PythonMLService {

    private final WebClient pythonWebClient;
    private static final Duration TIMEOUT = Duration.ofSeconds(120);

    /**
     * RAG问答
     */
    public Map<String, Object> askQuestion(QuestionRequest request) {
        log.info("调用Python RAG服务: {}", request.getQuestion());
        
        Map<String, Object> body = Map.of(
            "question", request.getQuestion(),
            "session_id", request.getSessionId() != null ? request.getSessionId() : "",
            "use_rewrite", request.isUseRewrite()
        );

        return pythonWebClient.post()
                .uri("/api/v1/ask")
                .bodyValue(body)
                .retrieve()
                .bodyToMono(Map.class)
                .block(TIMEOUT);
    }

    /**
     * Agent智能问答
     */
    public Map<String, Object> agentQuery(AgentRequest request) {
        log.info("调用Python Agent服务: {}", request.getQuery());
        
        Map<String, Object> body = Map.of(
            "query", request.getQuery(),
            "max_steps", request.getMaxSteps(),
            "verbose", request.isVerbose()
        );

        return pythonWebClient.post()
                .uri("/api/v1/agent")
                .bodyValue(body)
                .retrieve()
                .bodyToMono(Map.class)
                .block(TIMEOUT);
    }

    /**
     * 文献检索
     */
    public Map<String, Object> retrieve(RetrievalRequest request) {
        log.info("调用Python检索服务: {} ({})", request.getQuery(), request.getMethod());
        
        Map<String, Object> body = Map.of(
            "query", request.getQuery(),
            "top_k", request.getTopK(),
            "method", request.getMethod()
        );

        return pythonWebClient.post()
                .uri("/api/v1/retrieve")
                .bodyValue(body)
                .retrieve()
                .bodyToMono(Map.class)
                .block(TIMEOUT);
    }

    /**
     * 健康检查
     */
    public Map<String, Object> healthCheck() {
        return pythonWebClient.get()
                .uri("/health")
                .retrieve()
                .bodyToMono(Map.class)
                .block(Duration.ofSeconds(5));
    }

    /**
     * 系统统计
     */
    public Map<String, Object> getStats() {
        return pythonWebClient.get()
                .uri("/api/v1/stats")
                .retrieve()
                .bodyToMono(Map.class)
                .block(Duration.ofSeconds(10));
    }
}
