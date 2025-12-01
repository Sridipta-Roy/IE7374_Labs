import json
import socket
import logging
import os
from datetime import datetime
from pathlib import Path

class MLOpsLogger:
    def __init__(self, experiment_id, logstash_host='logstash', pipeline_port=5000, metrics_port=5001):
        self.experiment_id = experiment_id
        self.logstash_host = logstash_host
        self.pipeline_port = pipeline_port
        self.metrics_port = metrics_port
        
        # Setup file logging
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        self.file_logger = logging.getLogger(f"mlops_{experiment_id}")
        self.file_logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(log_dir / f"experiment_{experiment_id}.log")
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        self.file_logger.addHandler(handler)
    
    def send_to_logstash(self, log_data, port):
        """Send log data to Logstash via TCP"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect((self.logstash_host, port))
            
            log_json = json.dumps(log_data) + '\n'
            sock.sendall(log_json.encode('utf-8'))
            sock.close()
            return True
        except Exception as e:
            print(f"Error sending to Logstash: {e}")
            return False
    
    def log_pipeline_event(self, stage, event_type, data):
        """Log pipeline stage events"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "experiment_id": self.experiment_id,
            "stage": stage,
            "event_type": event_type,
            "type": "pipeline",
            **data
        }
        
        # Send to Logstash
        self.send_to_logstash(log_data, self.pipeline_port)
        
        # Write to file
        self.file_logger.info(json.dumps(log_data))
    
    def log_metrics(self, metrics):
        """Log final experiment metrics"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "experiment_id": self.experiment_id,
            "type": "metrics",
            **metrics
        }
        
        # Send to Logstash
        self.send_to_logstash(log_data, self.metrics_port)
        
        # Write to file
        self.file_logger.info(json.dumps(log_data))
    
    def log_error(self, stage, error_message, exception=None):
        """Log errors"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "experiment_id": self.experiment_id,
            "stage": stage,
            "event_type": "error",
            "type": "pipeline",
            "error_message": error_message,
            "exception": str(exception) if exception else None,
            "severity": "error"
        }
        
        self.send_to_logstash(log_data, self.pipeline_port)
        self.file_logger.error(json.dumps(log_data))