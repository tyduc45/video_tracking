# 生成无序列帧并在cuda设备上推理,视频源列表为【a.mp4,b.mp4】 a ,b均为相对路径,输出到result
- 实例：python main.py -i ../videos/video00.mp4 ../videos/video01.mp4 -o ../result -d cuda --no-frames