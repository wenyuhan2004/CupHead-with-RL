# test_image_fps.py - 图像处理性能测试
import time
import cv2
import numpy as np
import dxcam
import json

def test_screen_capture():
    """测试屏幕捕获性能"""
    print("=== 屏幕捕获测试 ===")
    
    cam = dxcam.create(output_color="BGR")
    cam.start(target_fps=60, video_mode=True)
    time.sleep(0.1)
    
    # 预热
    for _ in range(10):
        frame = cam.get_latest_frame()
    
    # 测试不同FPS设置
    for fps_setting in [30, 60, 120, 0]:  # 0表示无限制
        cam.stop()
        cam.start(target_fps=fps_setting, video_mode=True)
        time.sleep(0.1)
        
        frames_captured = 0
        start_time = time.time()
        test_duration = 3.0  # 测试3秒
        
        while time.time() - start_time < test_duration:
            frame = cam.get_latest_frame()
            if frame is not None:
                frames_captured += 1
        
        elapsed = time.time() - start_time
        fps = frames_captured / elapsed
        print(f"target_fps={fps_setting}: 实际FPS={fps:.2f}, 捕获帧数={frames_captured}")
    
    cam.stop()

def test_image_processing():
    """测试图像处理性能"""
    print("\n=== 图像处理测试 ===")
    
    # 模拟不同分辨率的输入
    resolutions = [
        (1920, 1080, "1080p"),
        (1280, 720, "720p"),
        (960, 540, "540p"),
        (640, 360, "360p")
    ]
    
    target_sizes = [
        (96, 96, "96x96"),
        (64, 64, "64x64"),
        (48, 48, "48x48"),
        (32, 32, "32x32")
    ]
    
    for width, height, res_name in resolutions:
        print(f"\n--- 输入分辨率: {res_name} ---")
        
        # 生成测试图像
        test_frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        for target_w, target_h, size_name in target_sizes:
            # 测试完整处理链路
            times = []
            for _ in range(100):
                start = time.time()
                
                # 1. 颜色转换
                gray = cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)
                
                # 2. 尺寸调整
                resized = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA)
                
                elapsed = time.time() - start
                times.append(elapsed * 1000)  # 转换为毫秒
            
            avg_time = np.mean(times)
            fps = 1000 / avg_time if avg_time > 0 else 0
            print(f"  {size_name}: {avg_time:.2f}ms/frame, {fps:.1f} FPS")

def test_interpolation_methods():
    """测试不同插值方法的性能"""
    print("\n=== 插值方法性能测试 ===")
    
    # 1080p -> 64x64
    test_frame = np.random.randint(0, 255, (1080, 1920), dtype=np.uint8)
    
    methods = [
        (cv2.INTER_AREA, "INTER_AREA"),
        (cv2.INTER_LINEAR, "INTER_LINEAR"),
        (cv2.INTER_NEAREST, "INTER_NEAREST"),
        (cv2.INTER_CUBIC, "INTER_CUBIC")
    ]
    
    for method, name in methods:
        times = []
        for _ in range(100):
            start = time.time()
            resized = cv2.resize(test_frame, (64, 64), interpolation=method)
            elapsed = time.time() - start
            times.append(elapsed * 1000)
        
        avg_time = np.mean(times)
        fps = 1000 / avg_time if avg_time > 0 else 0
        print(f"{name}: {avg_time:.2f}ms/frame, {fps:.1f} FPS")

def test_frame_stack_operations():
    """测试帧堆栈操作性能"""
    print("\n=== 帧堆栈操作测试 ===")
    
    sizes = [(32, 32), (48, 48), (64, 64), (96, 96)]
    stacks = [2, 3, 4, 5]
    
    for w, h in sizes:
        print(f"\n--- 帧大小: {w}x{h} ---")
        
        for stack_size in stacks:
            # 方法1: np.concatenate
            stackbuf1 = np.zeros((stack_size, h, w), dtype=np.uint8)
            new_frame = np.random.randint(0, 255, (h, w), dtype=np.uint8)
            
            times1 = []
            for _ in range(1000):
                start = time.time()
                stackbuf1 = np.concatenate([stackbuf1[1:], new_frame[None, ...]], axis=0)
                elapsed = time.time() - start
                times1.append(elapsed * 1000)
            
            # 方法2: 直接赋值
            stackbuf2 = np.zeros((stack_size, h, w), dtype=np.uint8)
            times2 = []
            for _ in range(1000):
                start = time.time()
                stackbuf2[:-1] = stackbuf2[1:]
                stackbuf2[-1] = new_frame
                elapsed = time.time() - start
                times2.append(elapsed * 1000)
            
            avg1 = np.mean(times1)
            avg2 = np.mean(times2)
            speedup = avg1 / avg2 if avg2 > 0 else 0
            
            print(f"  Stack={stack_size}: concatenate={avg1:.3f}ms, 直接赋值={avg2:.3f}ms, 加速={speedup:.1f}x")

def test_real_environment_simulation():
    """模拟真实环境的完整图像处理流程"""
    print("\n=== 真实环境模拟测试 ===")
    
    # 初始化相机
    cam = dxcam.create(output_color="BGR")
    cam.start(target_fps=60, video_mode=True)
    time.sleep(0.1)
    
    # 不同配置
    configs = [
        {"size": (96, 96), "stack": 4, "name": "原始配置"},
        {"size": (64, 64), "stack": 4, "name": "降低分辨率"},
        {"size": (48, 48), "stack": 4, "name": "更低分辨率"},
        {"size": (64, 64), "stack": 2, "name": "减少堆栈"},
        {"size": (32, 32), "stack": 2, "name": "极简配置"},
    ]
    
    for config in configs:
        w, h = config["size"]
        stack_size = config["stack"]
        name = config["name"]
        
        # 初始化帧堆栈
        stackbuf = np.zeros((stack_size, h, w), dtype=np.uint8)
        
        step_times = []
        frame_count = 0
        test_duration = 3.0
        start_time = time.time()
        
        while time.time() - start_time < test_duration:
            step_start = time.time()
            
            # 1. 捕获屏幕
            frame = cam.get_latest_frame()
            if frame is None:
                continue
            
            # 2. 图像处理
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            resized = cv2.resize(gray, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # 3. 帧堆栈更新
            stackbuf[:-1] = stackbuf[1:]
            stackbuf[-1] = resized
            
            step_elapsed = time.time() - step_start
            step_times.append(step_elapsed * 1000)
            frame_count += 1
        
        total_elapsed = time.time() - start_time
        avg_step_time = np.mean(step_times)
        fps = frame_count / total_elapsed
        theoretical_fps = 1000 / avg_step_time if avg_step_time > 0 else 0
        
        print(f"{name}: 实际FPS={fps:.1f}, 理论FPS={theoretical_fps:.1f}, 平均步时={avg_step_time:.2f}ms")
    
    cam.stop()

if __name__ == "__main__":
    print("Cuphead 图像处理性能测试")
    print("=" * 50)
    
    try:
        test_screen_capture()
        test_image_processing()
        test_interpolation_methods()
        test_frame_stack_operations()
        test_real_environment_simulation()
        
    except KeyboardInterrupt:
        print("\n测试中断")
    except Exception as e:
        print(f"\n测试出错: {e}")
    
    print("\n测试完成!")