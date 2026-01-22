"""
视频生成和可视化模块，将带追踪信息的帧序列转换为视频
"""
import cv2
import os
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple


class VideoVisualizer:
    """
    将带追踪信息的帧转换为视频文件
    """
    
    def __init__(self, output_dir: str, fps: int = 30, codec: str = 'mp4v'):
        """
        初始化视频生成器
        
        Args:
            output_dir: 输出目录
            fps: 帧率
            codec: 视频编码器
        """
        self.output_dir = output_dir
        self.fps = fps
        self.codec = codec
        self.frame_groups = defaultdict(list)  # 按视频源分组的帧
        
        os.makedirs(output_dir, exist_ok=True)
    
    def load_frame_directory(self, frame_dir: str):
        """
        从目录加载所有追踪帧，按视频源分组
        
        Args:
            frame_dir: 包含追踪帧的目录
        """
        frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(('.jpg', '.png'))])
        
        for frame_file in frame_files:
            # 解析文件名格式: tracked_*_video_X_fN_*.jpg
            if 'tracked_' in frame_file:
                parts = frame_file.split('_')
                # 提取视频 ID (video_0, video_1 等)
                video_id = None
                frame_id = None
                
                for i, part in enumerate(parts):
                    if part.startswith('video'):
                        video_id = part
                    if part.startswith('f') and part[1:].isdigit():
                        frame_id = int(part[1:])
                
                if video_id:
                    frame_path = os.path.join(frame_dir, frame_file)
                    self.frame_groups[video_id].append({
                        'path': frame_path,
                        'filename': frame_file,
                        'frame_id': frame_id if frame_id else float('inf')
                    })
        
        # 对每个视频源的帧按帧编号排序
        for video_id in self.frame_groups:
            self.frame_groups[video_id].sort(key=lambda x: x['frame_id'])
            print(f"[Visualizer] 加载了 {video_id} 的 {len(self.frame_groups[video_id])} 帧")
    
    def create_videos(self) -> List[str]:
        """
        为每个视频源生成追踪视频
        
        Returns:
            生成的视频文件路径列表
        """
        output_videos = []
        
        for video_id, frames in self.frame_groups.items():
            if not frames:
                continue
            
            try:
                # 读取第一帧以获取分辨率
                first_frame = cv2.imread(frames[0]['path'])
                if first_frame is None:
                    print(f"[Visualizer] 无法读取第一帧: {frames[0]['path']}")
                    continue
                
                height, width = first_frame.shape[:2]
                
                # 创建视频写入器
                output_path = os.path.join(self.output_dir, f"tracked_{video_id}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*self.codec)
                writer = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
                
                frame_count = 0
                for frame_info in frames:
                    frame = cv2.imread(frame_info['path'])
                    if frame is None:
                        print(f"[Visualizer] 无法读取帧: {frame_info['path']}")
                        continue
                    
                    # 在帧上添加文本信息（视频 ID 和帧编号）
                    text = f"{video_id} - Frame {frame_info['frame_id']}"
                    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (0, 255, 0), 2)
                    
                    writer.write(frame)
                    frame_count += 1
                
                writer.release()
                print(f"[Visualizer] 生成视频: {output_path} ({frame_count} 帧)")
                output_videos.append(output_path)
                
            except Exception as e:
                print(f"[Visualizer] 生成 {video_id} 的视频失败: {e}")
        
        return output_videos
    
    def create_comparison_video(self, original_dir: str = None) -> str:
        """
        创建对比视频（原始 + 追踪结果并排显示）
        
        Args:
            original_dir: 原始帧目录（可选）
            
        Returns:
            对比视频路径
        """
        if not self.frame_groups:
            print("[Visualizer] 没有追踪帧可用")
            return None
        
        # 获取第一个视频源的帧作为示例
        video_id = list(self.frame_groups.keys())[0]
        frames = self.frame_groups[video_id]
        
        if not frames:
            return None
        
        try:
            first_frame = cv2.imread(frames[0]['path'])
            height, width = first_frame.shape[:2]
            
            # 创建宽度为两倍的视频
            output_path = os.path.join(self.output_dir, f"comparison_{video_id}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            writer = cv2.VideoWriter(output_path, fourcc, self.fps, (width * 2, height))
            
            frame_count = 0
            for frame_info in frames:
                tracked_frame = cv2.imread(frame_info['path'])
                if tracked_frame is None:
                    continue
                
                # 创建并排图像
                combined = cv2.hconcat([tracked_frame, tracked_frame])
                
                writer.write(combined)
                frame_count += 1
            
            writer.release()
            print(f"[Visualizer] 生成对比视频: {output_path} ({frame_count} 帧)")
            return output_path
            
        except Exception as e:
            print(f"[Visualizer] 生成对比视频失败: {e}")
            return None


def visualize_results(frame_dir: str, output_dir: str = None, fps: int = 30):
    """
    便捷函数：将帧目录转换为视频
    
    Args:
        frame_dir: 包含追踪帧的目录
        output_dir: 输出目录（默认为 frame_dir/videos）
        fps: 帧率
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(frame_dir), "videos")
    
    visualizer = VideoVisualizer(output_dir, fps)
    visualizer.load_frame_directory(frame_dir)
    videos = visualizer.create_videos()
    
    print(f"\n=== 生成完成 ===")
    print(f"输出目录: {output_dir}")
    print(f"生成的视频:")
    for video in videos:
        print(f"  - {video}")
    
    return videos
