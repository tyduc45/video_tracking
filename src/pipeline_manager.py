"""
管理多条独立Pipeline的全局管理器
"""

import threading
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from video_source import VideoSource
from single_video_pipeline import SingleVideoPipeline
from pipeline_data import PipelineStatistics

logger = logging.getLogger(__name__)


class PipelineManager:
    """
    管理系统中的多条Pipeline
    
    职责:
    1. 创建和销毁Pipeline
    2. 启动/停止所有Pipeline
    3. 监控Pipeline状态
    4. 收集统计信息
    """
    
    def __init__(self, output_dir: str = "result", max_pipelines: int = 10):
        """
        初始化Manager
        
        Args:
            output_dir: 所有Pipeline的输出目录
            max_pipelines: 最多允许创建的Pipeline数量
        """
        self.output_dir = output_dir
        self.max_pipelines = max_pipelines
        
        # 存储所有Pipeline
        self.pipelines: Dict[str, SingleVideoPipeline] = {}
        self.pipeline_counter = 0
        
        # 线程安全
        self.lock = threading.Lock()
        
        logger.info(f"PipelineManager initialized (output_dir={output_dir}, max={max_pipelines})")
    
    def create_pipeline(self, video_source: VideoSource,
                       inference_func = None,
                       tracker_instance = None,
                       save_func = None,
                       pipeline_id: Optional[str] = None) -> str:
        """
        创建一条新的Pipeline
        
        Args:
            video_source: VideoSource实例
            inference_func: 推理函数
            tracker_instance: 追踪器实例
            save_func: 保存函数
            pipeline_id: Pipeline ID（若None则自动生成）
        
        Returns:
            str: Pipeline ID
        """
        with self.lock:
            # 检查是否超过限制
            if len(self.pipelines) >= self.max_pipelines:
                raise RuntimeError(f"Cannot create more than {self.max_pipelines} pipelines")
            
            # 生成Pipeline ID
            if pipeline_id is None:
                pipeline_id = f"video_{self.pipeline_counter}"
                self.pipeline_counter += 1
            
            # 检查ID是否已存在
            if pipeline_id in self.pipelines:
                raise ValueError(f"Pipeline ID '{pipeline_id}' already exists")
            
            # 创建Pipeline
            pipeline = SingleVideoPipeline(
                pipeline_id=pipeline_id,
                video_source=video_source,
                inference_func=inference_func,
                tracker_instance=tracker_instance,
                save_func=save_func,
                output_dir=self.output_dir
            )
            
            self.pipelines[pipeline_id] = pipeline
            
            logger.info(f"Created pipeline: {pipeline_id} (source={video_source.name})")
            
            return pipeline_id
    
    def start_all(self):
        """启动所有Pipeline"""
        with self.lock:
            if not self.pipelines:
                logger.warning("No pipelines to start")
                return
            
            logger.info(f"Starting {len(self.pipelines)} pipelines...")
            
            for pipeline_id, pipeline in self.pipelines.items():
                try:
                    pipeline.start()
                except Exception as e:
                    logger.error(f"Failed to start pipeline {pipeline_id}: {e}")
    
    def stop_all(self):
        """停止所有Pipeline"""
        with self.lock:
            logger.info(f"Stopping {len(self.pipelines)} pipelines...")
            
            for pipeline_id, pipeline in self.pipelines.items():
                try:
                    pipeline.stop()
                except Exception as e:
                    logger.error(f"Failed to stop pipeline {pipeline_id}: {e}")
    
    def wait_all(self, timeout: Optional[float] = None) -> bool:
        """
        等待所有Pipeline完成
        
        Args:
            timeout: 超时时间（秒）
        
        Returns:
            bool: 是否全部完成
        """
        if not self.pipelines:
            logger.warning("No pipelines to wait for")
            return True
        
        logger.info(f"Waiting for {len(self.pipelines)} pipelines to complete...")
        
        all_done = True
        
        with self.lock:
            for pipeline_id, pipeline in self.pipelines.items():
                if not pipeline.wait(timeout=timeout):
                    all_done = False
                    logger.warning(f"Pipeline {pipeline_id} did not complete within timeout")
        
        if all_done:
            logger.info("All pipelines completed")
        else:
            logger.warning("Some pipelines did not complete within timeout")
        
        return all_done
    
    def get_pipeline(self, pipeline_id: str) -> Optional[SingleVideoPipeline]:
        """获取指定的Pipeline"""
        return self.pipelines.get(pipeline_id)
    
    def get_all_pipelines(self) -> List[Tuple[str, SingleVideoPipeline]]:
        """获取所有Pipeline"""
        with self.lock:
            return list(self.pipelines.items())
    
    def get_status(self) -> Dict:
        """
        获取所有Pipeline的状态
        
        Returns:
            dict: 包含各Pipeline的状态信息
        """
        status = {
            'timestamp': datetime.now().isoformat(),
            'total_pipelines': len(self.pipelines),
            'pipelines': {}
        }
        
        with self.lock:
            for pipeline_id, pipeline in self.pipelines.items():
                stats = pipeline.get_statistics()
                status['pipelines'][pipeline_id] = {
                    'state': stats.processing_state,
                    'total_frames': stats.total_frames,
                    'dropped_frames': stats.dropped_frames,
                    'average_inference_fps': f"{stats.average_inference_fps:.1f}",
                    'average_tracking_fps': f"{stats.average_tracking_fps:.1f}",
                    'error': stats.error_message
                }
        
        return status
    
    def print_status(self):
        """打印所有Pipeline的状态"""
        status = self.get_status()
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Pipeline Manager Status")
        logger.info(f"{'='*70}")
        logger.info(f"Timestamp:        {status['timestamp']}")
        logger.info(f"Total Pipelines:  {status['total_pipelines']}")
        logger.info(f"{'-'*70}")
        
        for pipeline_id, pipeline_info in status['pipelines'].items():
            logger.info(f"{pipeline_id}:")
            logger.info(f"  State:         {pipeline_info['state']}")
            logger.info(f"  Frames:        {pipeline_info['total_frames']} (dropped: {pipeline_info['dropped_frames']})")
            logger.info(f"  Inference FPS: {pipeline_info['average_inference_fps']}")
            logger.info(f"  Tracking FPS:  {pipeline_info['average_tracking_fps']}")
            if pipeline_info['error']:
                logger.info(f"  Error:         {pipeline_info['error']}")
        
        logger.info(f"{'='*70}\n")
    
    def print_all_statistics(self):
        """打印所有Pipeline的详细统计信息"""
        logger.info(f"\n{'='*70}")
        logger.info(f"All Pipeline Statistics")
        logger.info(f"{'='*70}\n")
        
        with self.lock:
            for pipeline_id, pipeline in self.pipelines.items():
                pipeline.print_statistics()
    
    def reset(self):
        """清空所有Pipeline"""
        with self.lock:
            logger.warning("Resetting PipelineManager (clearing all pipelines)")
            self.pipelines.clear()
            self.pipeline_counter = 0
