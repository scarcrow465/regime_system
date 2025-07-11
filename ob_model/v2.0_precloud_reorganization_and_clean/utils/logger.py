#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Logging utilities for the Regime System
Provides consistent logging across all modules
"""

import logging
import sys
import os
from datetime import datetime
from typing import Optional, Dict, Any
import json
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from tqdm import tqdm  # For visual progress bars

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import LOG_DIR, LOG_LEVEL, LOG_FORMAT

# =============================================================================
# CUSTOM FORMATTERS
# =============================================================================

class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
        
        formatted = super().format(record)
        
        record.levelname = levelname
        
        return formatted

class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
        }
        
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'msecs', 'levelname', 
                          'levelno', 'pathname', 'filename', 'module', 'funcName', 
                          'lineno', 'exc_info', 'exc_text', 'stack_info', 'message']:
                log_data[key] = value
        
        return json.dumps(log_data)

# =============================================================================
# LOGGER SETUP
# =============================================================================

def setup_logger(name: str = 'regime_system',
                level: Optional[str] = None,
                log_file: Optional[str] = None,
                console: bool = True,
                file_logging: bool = True,
                json_format: bool = False) -> logging.Logger:
    """
    Setup a logger with console and file handlers. Uses date_time (yyyy-mm-dd_hh-mm-ss) for new file per run—no overload.
    """
    logger = logging.getLogger(name)
    
    level = level or LOG_LEVEL
    logger.setLevel(getattr(logging, level.upper()))
    
    logger.handlers = []
    
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        
        if json_format:
            console_formatter = JsonFormatter()
        else:
            console_formatter = ColoredFormatter(LOG_FORMAT)
        
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    if file_logging:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        module_dir = os.path.join(LOG_DIR, name.lower())
        os.makedirs(module_dir, exist_ok=True)
        
        if log_file is None:
            log_file = os.path.join(module_dir, f'{name}_{timestamp}.log')
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,
            backupCount=5
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        
        if json_format:
            file_formatter = JsonFormatter()
        else:
            file_formatter = logging.Formatter(LOG_FORMAT)
        
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

# =============================================================================
# SPECIALIZED LOGGERS
# =============================================================================

class PerformanceLogger:
    """Logger for tracking performance metrics"""
    
    def __init__(self, name: str = 'performance'):
        self.logger = setup_logger(f'regime_system.{name}', json_format=True)
        self.metrics = {}
        
    def log_metric(self, metric_name: str, value: float, 
                  context: Optional[Dict[str, Any]] = None):
        """Log a performance metric"""
        log_data = {
            'metric_name': metric_name,
            'value': value,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if context:
            log_data['context'] = context
        
        self.logger.info("Performance metric", extra=log_data)
        
        # Store for aggregation
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)
    
    def log_optimization_iteration(self, iteration: int, score: float,
                                 params: Dict[str, float], metrics: Dict[str, float]):
        """Log optimization iteration details"""
        self.logger.info("Optimization iteration", extra={
            'iteration': iteration,
            'score': score,
            'parameters': params,
            'metrics': metrics
        })
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics"""
        summary = {}
        
        for metric_name, values in self.metrics.items():
            if values:
                import numpy as np
                summary[metric_name] = {
                    'count': len(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'last': values[-1]
                }
        
        return summary

class TradingLogger:
    """Logger for trading operations"""
    
    def __init__(self, name: str = 'trading'):
        self.logger = setup_logger(f'regime_system.{name}')
        
    def log_trade(self, action: str, symbol: str, quantity: float, 
                 price: float, reason: str, **kwargs):
        """Log a trade execution"""
        self.logger.info(f"Trade: {action} {quantity} {symbol} @ {price} - {reason}", 
                        extra={
                            'action': action,
                            'symbol': symbol,
                            'quantity': quantity,
                            'price': price,
                            'reason': reason,
                            **kwargs
                        })
    
    def log_position_update(self, symbol: str, position: float, 
                          market_value: float, pnl: float):
        """Log position update"""
        self.logger.info(f"Position: {symbol} = {position} (${market_value:.2f}, "
                        f"P&L: ${pnl:.2f})", 
                        extra={
                            'symbol': symbol,
                            'position': position,
                            'market_value': market_value,
                            'pnl': pnl
                        })
    
    def log_risk_alert(self, alert_type: str, message: str, **kwargs):
        """Log risk management alert"""
        self.logger.warning(f"Risk Alert [{alert_type}]: {message}", 
                           extra={
                               'alert_type': alert_type,
                               **kwargs
                           })

# =============================================================================
# LOGGING DECORATORS
# =============================================================================

# =============================================================================
# LOGGING DECORATORS (Enhanced with tqdm for bars)
# =============================================================================

def log_execution_time(logger: Optional[logging.Logger] = None):
    """Decorator to log function execution time"""
    import functools
    import time
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = logging.getLogger(func.__module__)
            
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                logger.info(f"{func.__name__} completed in {execution_time:.2f}s")
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"{func.__name__} failed after {execution_time:.2f}s: {e}")
                raise
        
        return wrapper
    return decorator

def log_errors(logger: Optional[logging.Logger] = None, 
              reraise: bool = True):
    """Decorator to log exceptions"""
    import functools
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = logging.getLogger(func.__module__)
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
                if reraise:
                    raise
                return None
        
        return wrapper
    return decorator

# =============================================================================
# LOG ANALYSIS UTILITIES
# =============================================================================

class LogAnalyzer:
    """Analyze log files for patterns and issues"""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        
    def analyze_errors(self) -> Dict[str, Any]:
        """Analyze error patterns in logs"""
        errors = []
        error_counts = {}
        
        with open(self.log_file, 'r') as f:
            for line in f:
                if 'ERROR' in line or 'CRITICAL' in line:
                    errors.append(line.strip())
                    
                    # Try to extract error type
                    if 'Error' in line:
                        error_type = line.split('Error')[0].split()[-1] + 'Error'
                        error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        return {
            'total_errors': len(errors),
            'error_types': error_counts,
            'recent_errors': errors[-10:] if errors else []
        }
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze performance metrics from logs"""
        execution_times = []
        
        with open(self.log_file, 'r') as f:
            for line in f:
                if 'completed in' in line and 's' in line:
                    try:
                        # Extract execution time
                        time_str = line.split('completed in')[1].split('s')[0].strip()
                        execution_times.append(float(time_str))
                    except:
                        pass
        
        if execution_times:
            import numpy as np
            return {
                'total_operations': len(execution_times),
                'avg_execution_time': np.mean(execution_times),
                'max_execution_time': np.max(execution_times),
                'min_execution_time': np.min(execution_times),
                'total_time': np.sum(execution_times)
            }
        
        return {}

# =============================================================================
# GLOBAL LOGGER INSTANCES
# =============================================================================

# Create default loggers
main_logger = setup_logger('regime_system')
performance_logger = PerformanceLogger()
trading_logger = TradingLogger()

# Export convenience functions
def get_logger(name: str) -> logging.Logger:
    """Get a logger instance"""
    return setup_logger(name)

def log_info(message: str, **kwargs):
    """Log info message"""
    main_logger.info(message, extra=kwargs)

def log_error(message: str, **kwargs):
    """Log error message"""
    main_logger.error(message, extra=kwargs)

def log_warning(message: str, **kwargs):
    """Log warning message"""
    main_logger.warning(message, extra=kwargs)

def log_debug(message: str, **kwargs):
    """Log debug message"""
    main_logger.debug(message, extra=kwargs)

def progress_wrapper(iterable, desc="Progress", logger=main_logger, level='INFO'):
    """
    Wraps iterable with tqdm bar for visual progress in terminal (INFO level).
    - Why: Makes long runs feel "alive" and trackable, like a video game loading screen—your visual style.
    - Use: for item in progress_wrapper(range(100), desc="Scanning edges"):
    """
    if logger.level <= logging.INFO:
        return tqdm(iterable, desc=desc, file=sys.stdout)  # Visual bar in terminal
    return iterable  # No bar if higher level

