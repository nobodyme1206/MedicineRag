package com.medical.rag.controller;

import com.medical.rag.dto.AgentRequest;
import com.medical.rag.dto.QuestionRequest;
import com.medical.rag.dto.RetrievalRequest;
import com.medical.rag.service.PythonMLService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@Slf4j
@RestController
@RequestMapping("/api/v1")
@RequiredArgsConstructor
@Tag(name = "医学RAG API", description = "医学知识问答与检索接口")
public class RagController {

    private final PythonMLService pythonMLService;

    @PostMapping("/ask")
    @Operation(summary = "RAG问答", description = "基于检索增强生成的医学问答")
    public ResponseEntity<Map<String, Object>> ask(@Valid @RequestBody QuestionRequest request) {
        log.info("收到RAG问答请求: {}", request.getQuestion());
        Map<String, Object> result = pythonMLService.askQuestion(request);
        return ResponseEntity.ok(result);
    }

    @PostMapping("/agent")
    @Operation(summary = "Agent智能问答", description = "ReAct模式智能代理问答")
    public ResponseEntity<Map<String, Object>> agent(@Valid @RequestBody AgentRequest request) {
        log.info("收到Agent请求: {}", request.getQuery());
        Map<String, Object> result = pythonMLService.agentQuery(request);
        return ResponseEntity.ok(result);
    }

    @PostMapping("/retrieve")
    @Operation(summary = "文献检索", description = "支持BM25、向量、混合检索")
    public ResponseEntity<Map<String, Object>> retrieve(@Valid @RequestBody RetrievalRequest request) {
        log.info("收到检索请求: {} ({})", request.getQuery(), request.getMethod());
        Map<String, Object> result = pythonMLService.retrieve(request);
        return ResponseEntity.ok(result);
    }

    @GetMapping("/stats")
    @Operation(summary = "系统统计", description = "获取系统状态统计信息")
    public ResponseEntity<Map<String, Object>> stats() {
        Map<String, Object> result = pythonMLService.getStats();
        return ResponseEntity.ok(result);
    }
}
