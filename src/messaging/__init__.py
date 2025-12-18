# -*- coding: utf-8 -*-
"""
消息队列模块 - Kafka集成
"""

from .kafka_producer import KafkaArticleProducer
from .kafka_consumer import KafkaArticleConsumer

__all__ = ['KafkaArticleProducer', 'KafkaArticleConsumer']
