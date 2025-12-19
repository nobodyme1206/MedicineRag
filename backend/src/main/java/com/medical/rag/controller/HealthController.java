package com.medical.rag.controller;

import com.medical.rag.service.PythonMLService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.HashMap;
import java.util.Map;

@Slf4j
@RestController
@RequiredArgsConstructor
@Tag(name = "健康检查", description = "系统健康状态检查")
public class HealthController {

    private final PythonMLService pythonMLService;

    @GetMapping("/health")
    @Operation(summary = "健康检查", description = "检查Java后端和Python ML服务状态")
    public ResponseEntity<Map<String, Object>> health() {
        Map<String, Object> result = new HashMap<>();
        result.put("java_backend", "healthy");
        
        try {
            Map<String, Object> pythonHealth = pythonMLService.healthCheck();
            result.put("python_ml_service", pythonHealth);
            result.put("status", "healthy");
        } catch (Exception e) {
            log.warn("Python ML服务不可用: {}", e.getMessage());
            result.put("python_ml_service", "unavailable");
            result.put("status", "degraded");
        }
        
        return ResponseEntity.ok(result);
    }
}
