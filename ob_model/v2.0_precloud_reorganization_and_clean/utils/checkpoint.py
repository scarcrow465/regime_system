#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Checkpoint functionality for saving and resuming optimization state
Essential for cloud deployment to avoid losing progress
"""

import pickle
import json
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import RESULTS_DIR

logger = logging.getLogger(__name__)

# =============================================================================
# CHECKPOINT MANAGER
# =============================================================================

class OptimizationCheckpoint:
    """
    Manages saving and loading optimization state
    Enables resuming interrupted optimizations
    """
    
    def __init__(self, checkpoint_dir: Optional[str] = None):
        """
        Initialize checkpoint manager
        
        Args:
            checkpoint_dir: Directory for checkpoint files
        """
        self.checkpoint_dir = checkpoint_dir or os.path.join(RESULTS_DIR, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
    
    def save_state(self, 
                   optimizer_state: Dict[str, Any],
                   iteration: int,
                   checkpoint_name: Optional[str] = None) -> str:
        """
        Save current optimization state
        
        Args:
            optimizer_state: Complete state dictionary
            iteration: Current iteration number
            checkpoint_name: Optional custom name
            
        Returns:
            Path to saved checkpoint
        """
        try:
            # Generate checkpoint name
            if checkpoint_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint_name = f"checkpoint_iter{iteration}_{timestamp}"
            
            # Prepare state for saving
            save_state = {
                'iteration': iteration,
                'timestamp': datetime.now().isoformat(),
                'optimizer_state': self._prepare_state_for_save(optimizer_state)
            }
            
            # Save as pickle (preserves numpy arrays and complex objects)
            checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}.pkl")
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(save_state, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Also save metadata as JSON for easy inspection
            metadata = {
                'iteration': iteration,
                'timestamp': save_state['timestamp'],
                'checkpoint_path': checkpoint_path,
                'state_keys': list(optimizer_state.keys())
            }
            
            metadata_path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            return checkpoint_path
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def load_state(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load optimization state from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Restored state dictionary
        """
        try:
            logger.info(f"Loading checkpoint: {checkpoint_path}")
            
            with open(checkpoint_path, 'rb') as f:
                saved_state = pickle.load(f)
            
            # Restore state
            iteration = saved_state['iteration']
            optimizer_state = self._restore_state_from_save(saved_state['optimizer_state'])
            
            logger.info(f"Checkpoint loaded: Iteration {iteration}")
            
            return {
                'iteration': iteration,
                'optimizer_state': optimizer_state,
                'timestamp': saved_state['timestamp']
            }
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def _prepare_state_for_save(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare state for pickling"""
        prepared_state = {}
        
        for key, value in state.items():
            if isinstance(value, pd.DataFrame):
                # Convert DataFrames to dict format
                prepared_state[key] = {
                    'type': 'dataframe',
                    'data': value.to_dict('records'),
                    'index': value.index.tolist(),
                    'columns': value.columns.tolist()
                }
            elif isinstance(value, pd.Series):
                # Convert Series to dict format
                prepared_state[key] = {
                    'type': 'series',
                    'data': value.to_dict(),
                    'index': value.index.tolist()
                }
            elif isinstance(value, np.ndarray):
                # Convert numpy arrays
                prepared_state[key] = {
                    'type': 'ndarray',
                    'data': value.tolist(),
                    'shape': value.shape,
                    'dtype': str(value.dtype)
                }
            else:
                # Keep as is for basic types
                prepared_state[key] = value
        
        return prepared_state
    
    def _restore_state_from_save(self, saved_state: Dict[str, Any]) -> Dict[str, Any]:
        """Restore state from saved format"""
        restored_state = {}
        
        for key, value in saved_state.items():
            if isinstance(value, dict) and 'type' in value:
                if value['type'] == 'dataframe':
                    # Restore DataFrame
                    df = pd.DataFrame(value['data'])
                    if value['index']:
                        df.index = value['index']
                    restored_state[key] = df
                
                elif value['type'] == 'series':
                    # Restore Series
                    restored_state[key] = pd.Series(value['data'])
                
                elif value['type'] == 'ndarray':
                    # Restore numpy array
                    arr = np.array(value['data'])
                    if value['shape']:
                        arr = arr.reshape(value['shape'])
                    restored_state[key] = arr
                else:
                    restored_state[key] = value
            else:
                restored_state[key] = value
        
        return restored_state
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints"""
        checkpoints = []
        
        for file in os.listdir(self.checkpoint_dir):
            if file.endswith('_metadata.json'):
                metadata_path = os.path.join(self.checkpoint_dir, file)
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    checkpoints.append(metadata)
                except Exception as e:
                    logger.warning(f"Failed to read metadata {file}: {e}")
        
        # Sort by timestamp
        checkpoints.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return checkpoints
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to most recent checkpoint"""
        checkpoints = self.list_checkpoints()
        
        if checkpoints:
            return checkpoints[0]['checkpoint_path']
        else:
            return None
    
    def cleanup_old_checkpoints(self, keep_last: int = 5):
        """Remove old checkpoints, keeping only the most recent ones"""
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) > keep_last:
            # Remove older checkpoints
            for checkpoint in checkpoints[keep_last:]:
                try:
                    # Remove pickle file
                    if os.path.exists(checkpoint['checkpoint_path']):
                        os.remove(checkpoint['checkpoint_path'])
                    
                    # Remove metadata file
                    metadata_path = checkpoint['checkpoint_path'].replace('.pkl', '_metadata.json')
                    if os.path.exists(metadata_path):
                        os.remove(metadata_path)
                    
                    logger.info(f"Removed old checkpoint: {checkpoint['checkpoint_path']}")
                    
                except Exception as e:
                    logger.warning(f"Failed to remove checkpoint: {e}")

# =============================================================================
# OPTIMIZATION STATE MANAGER
# =============================================================================

class OptimizationStateManager:
    """
    High-level manager for optimization state
    Handles automatic checkpointing during optimization
    """
    
    def __init__(self, 
                 checkpoint_interval: int = 10,
                 auto_checkpoint: bool = True):
        """
        Initialize state manager
        
        Args:
            checkpoint_interval: Save checkpoint every N iterations
            auto_checkpoint: Whether to automatically save checkpoints
        """
        self.checkpoint_manager = OptimizationCheckpoint()
        self.checkpoint_interval = checkpoint_interval
        self.auto_checkpoint = auto_checkpoint
        self.current_state = {}
        self.iteration_count = 0
        
    def update_state(self, 
                    iteration: int,
                    best_params: Dict[str, float],
                    best_score: float,
                    history: List[Dict],
                    additional_data: Optional[Dict] = None):
        """
        Update current optimization state
        
        Args:
            iteration: Current iteration
            best_params: Best parameters found so far
            best_score: Best score achieved
            history: Optimization history
            additional_data: Any additional data to save
        """
        self.iteration_count = iteration
        
        self.current_state = {
            'iteration': iteration,
            'best_params': best_params,
            'best_score': best_score,
            'optimization_history': history,
            'timestamp': datetime.now().isoformat()
        }
        
        if additional_data:
            self.current_state.update(additional_data)
        
        # Auto checkpoint if enabled
        if self.auto_checkpoint and iteration % self.checkpoint_interval == 0:
            self.save_checkpoint()
    
    def save_checkpoint(self, checkpoint_name: Optional[str] = None) -> str:
        """Save current state as checkpoint"""
        return self.checkpoint_manager.save_state(
            self.current_state,
            self.iteration_count,
            checkpoint_name
        )
    
    def resume_from_checkpoint(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Resume from checkpoint
        
        Args:
            checkpoint_path: Specific checkpoint to load, or None for latest
            
        Returns:
            Restored state
        """
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_manager.get_latest_checkpoint()
            
        if checkpoint_path is None:
            logger.warning("No checkpoint found to resume from")
            return None
        
        restored = self.checkpoint_manager.load_state(checkpoint_path)
        self.current_state = restored['optimizer_state']
        self.iteration_count = restored['iteration']
        
        logger.info(f"Resumed from iteration {self.iteration_count}")
        
        return self.current_state
    
    def get_resume_info(self) -> Dict[str, Any]:
        """Get information about available resume points"""
        checkpoints = self.checkpoint_manager.list_checkpoints()
        
        if not checkpoints:
            return {
                'can_resume': False,
                'latest_checkpoint': None,
                'available_checkpoints': []
            }
        
        latest = checkpoints[0]
        
        return {
            'can_resume': True,
            'latest_checkpoint': latest,
            'latest_iteration': latest['iteration'],
            'available_checkpoints': len(checkpoints),
            'checkpoints': checkpoints[:5]  # Show last 5
        }

# =============================================================================
# CLOUD COST MONITOR
# =============================================================================

class CloudCostMonitor:
    """
    Monitor cloud costs during optimization
    Implements early stopping based on cost thresholds
    """
    
    def __init__(self, 
                 max_cost_usd: float = 50.0,
                 cost_per_hour: float = 0.50):
        """
        Initialize cost monitor
        
        Args:
            max_cost_usd: Maximum allowed cost
            cost_per_hour: Estimated cost per hour of compute
        """
        self.max_cost_usd = max_cost_usd
        self.cost_per_hour = cost_per_hour
        self.start_time = datetime.now()
        self.checkpoints = []
        
    def update(self, iteration: int, current_score: float):
        """Update cost tracking"""
        elapsed_hours = (datetime.now() - self.start_time).total_seconds() / 3600
        current_cost = elapsed_hours * self.cost_per_hour
        
        self.checkpoints.append({
            'iteration': iteration,
            'elapsed_hours': elapsed_hours,
            'cost_usd': current_cost,
            'score': current_score,
            'timestamp': datetime.now().isoformat()
        })
        
        return current_cost
    
    def should_stop(self, iteration: int, current_score: float) -> Tuple[bool, str]:
        """
        Check if optimization should stop based on cost
        
        Returns:
            (should_stop, reason)
        """
        current_cost = self.update(iteration, current_score)
        
        # Check absolute cost limit
        if current_cost >= self.max_cost_usd:
            return True, f"Cost limit reached: ${current_cost:.2f} >= ${self.max_cost_usd:.2f}"
        
        # Check if we're close to limit
        if current_cost >= self.max_cost_usd * 0.9:
            logger.warning(f"Approaching cost limit: ${current_cost:.2f} / ${self.max_cost_usd:.2f}")
        
        # Check convergence (no improvement in last N iterations)
        if len(self.checkpoints) >= 20:
            recent_scores = [cp['score'] for cp in self.checkpoints[-20:]]
            score_std = np.std(recent_scores)
            
            if score_std < 0.001:  # Very little variation
                return True, f"Converged - score variation < 0.001 over last 20 iterations"
        
        return False, ""
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost summary"""
        if not self.checkpoints:
            return {}
        
        latest = self.checkpoints[-1]
        
        return {
            'total_cost_usd': latest['cost_usd'],
            'elapsed_hours': latest['elapsed_hours'],
            'cost_per_iteration': latest['cost_usd'] / latest['iteration'] if latest['iteration'] > 0 else 0,
            'iterations_completed': latest['iteration'],
            'budget_used_pct': (latest['cost_usd'] / self.max_cost_usd) * 100,
            'estimated_remaining_iterations': int((self.max_cost_usd - latest['cost_usd']) / 
                                                (latest['cost_usd'] / latest['iteration']))
                                               if latest['iteration'] > 0 else 0
        }
    
    def save_cost_report(self, filepath: Optional[str] = None):
        """Save detailed cost report"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(RESULTS_DIR, f"cost_report_{timestamp}.csv")
        
        df = pd.DataFrame(self.checkpoints)
        df.to_csv(filepath, index=False)
        
        logger.info(f"Cost report saved to: {filepath}")

