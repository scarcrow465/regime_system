#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Logging utilities for the Regime System
Provides consistent logging across all modules
"""

from loguru import logger as loguru_logger
import sys
import os
from datetime import datetime
from typing import Optional, Dict, Any
from tqdm import tqdm
from config.settings import LOG_DIR, LOG_LEVEL
import pretty_errors  # Prettify tracebacks globally for clearer errors

pretty_errors.configure(
    separator_character='*',  # Pretty separator
    filename_display=pretty_errors.FILENAME_EXTENDED,  # Show full file path
    line_number_first=True,  # Line number first for quick scan
    lines_before=5, lines_after=2,  # Context lines
)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Global Loguru setup (pretty console, deep file)
loguru_logger.remove()  # Clear defaults
loguru_logger.add(sys.stdout, level=LOG_LEVEL, colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")  # Pretty console
loguru_logger.add(os.path.join(LOG_DIR, "{time:YYYY-MM-DD_HH-MM-SS}.log"), level="DEBUG", rotation="10 MB")  # Deep file with timed naming and rotation

def get_logger(name: str) -> "loguru_logger":
    """Get a Loguru logger instance with name binding"""
    return loguru_logger.bind(name=name)

# =============================================================================
# SPECIALIZED LOGGERS (Adapted for Loguru)
# =============================================================================

class PerformanceLogger:
    """Logger for tracking performance metrics"""
    
    def __init__(self, name: str = 'performance'):
        self.logger = get_logger(f'regime_system.{name}')
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
        
        self.logger.info("Performance metric: {data}", data=log_data)
        
        # Store for aggregation
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)
    
    def log_optimization_iteration(self, iteration: int, score: float,
                                 params: Dict[str, float], metrics: Dict[str, float]):
        """Log optimization iteration details"""
        self.logger.info("Optimization iteration: iteration={iteration}, score={score}, parameters={params}, metrics={metrics}",
                         iteration=iteration, score=score, params=params, metrics=metrics)
    
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
        self.logger = get_logger(f'regime_system.{name}')
        
    def log_trade(self, action: str, symbol: str, quantity: float, 
                 price: float, reason: str, **kwargs):
        """Log a trade execution"""
        self.logger.info("Trade: {action} {quantity} {symbol} @ {price} - {reason}", 
                         action=action, quantity=quantity, symbol=symbol, price=price, reason=reason, **kwargs)
    
    def log_position_update(self, symbol: str, position: float, 
                          market_value: float, pnl: float):
        """Log position update"""
        self.logger.info("Position: {symbol} = {position} (${market_value:.2f}, P&L: ${pnl:.2f})", 
                         symbol=symbol, position=position, market_value=market_value, pnl=pnl)
    
    def log_risk_alert(self, alert_type: str, message: str, **kwargs):
        """Log risk management alert"""
        self.logger.warning("Risk Alert [{alert_type}]: {message}", 
                            alert_type=alert_type, message=message, **kwargs)

# =============================================================================
# LOGGING DECORATORS (Adapted for Loguru)
# =============================================================================

def log_execution_time(logger: Optional["loguru_logger"] = None):
    """Decorator to log function execution time"""
    import functools
    import time
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)
            
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                logger.info("{func_name} completed in {time:.2f}s", func_name=func.__name__, time=execution_time)
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error("{func_name} failed after {time:.2f}s: {error}", func_name=func.__name__, time=execution_time, error=e)
                raise
        
        return wrapper
    return decorator

def log_errors(logger: Optional["loguru_logger"] = None, 
              reraise: bool = True):
    """Decorator to log exceptions"""
    import functools
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error("Error in {func_name}: {error}", func_name=func.__name__, error=e)
                if reraise:
                    raise
                return None
        
        return wrapper
    return decorator

# =============================================================================
# LOG ANALYSIS UTILITIES (Adapted for Loguru format)
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
                    
                    # Try to extract error type (adjusted for Loguru format)
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
                        # Extract execution time (adjusted for Loguru format)
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
main_logger = get_logger('regime_system')
performance_logger = PerformanceLogger()
trading_logger = TradingLogger()

# Export convenience functions (use Loguru syntax)
def log_info(message: str, **kwargs):
    """Log info message"""
    main_logger.info(message, **kwargs)

def log_error(message: str, **kwargs):
    """Log error message"""
    main_logger.error(message, **kwargs)

def log_warning(message: str, **kwargs):
    """Log warning message"""
    main_logger.warning(message, **kwargs)

def log_debug(message: str, **kwargs):
    """Log debug message"""
    main_logger.debug(message, **kwargs)

def progress_wrapper(iterable, desc="Progress", logger=main_logger, level='INFO'):
    """
    Wraps iterable with tqdm bar for visual progress in terminal (INFO level).
    - Why: Makes long runs feel "alive" and trackable, like a video game loading screen—your visual style.
    - Use: for item in progress_wrapper(range(100), desc="Scanning edges"):
    """
    if logger.level(level) <= logger.level("INFO"):
        return tqdm(iterable, desc=desc, file=sys.stdout)  # Visual bar in terminal
    return iterable  # No bar if higher level

