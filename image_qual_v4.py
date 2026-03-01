import os
import gc
import sys
import cv2
import torch
import subprocess
import argparse
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from tqdm import tqdm
import vapoursynth as vs
import havsfunc
from datetime import datetime

core = vs.core
# プラグインパスは環境に合わせて自動読み込み
plugin_path = r"C:\Users\cammy\AppData\Roaming\VapourSynth\plugins64"
if os.path.exists(plugin_path):
    try: core.std.LoadAllPlugins(plugin_path)
    except Exception as e: print(f"⚠️ プラグイン警告: {e}")

def process_final_video(video_in, audio_src, final_out):
    """FFmpegで2Kキャンバス化とシャープネス処理"""
    print(f"\n--- 最終エンコード (2K + Unsharp) ---")
    filter_str = (
        "scale=1920:1440:flags=lanczos,"
        "null,"
        "pad=2560:1440:(ow-iw)/2:(oh-ih)/2:black"
    )
    cmd = [
        'ffmpeg', '-y', '-i', video_in, '-i', audio_src, 
        '-vf', filter_str, '-map', '0:v', '-map', '1:a:0', 
        '-c:v', 'libx265', '-crf', '18', '-preset', 'slow',
        '-c:a', 'aac', '-b:a', '256k', final_out
    ]
    subprocess.run(cmd, check=True)

def main():
    start_time = datetime.now()
    parser = argparse.ArgumentParser(description="LD Restoration: Real-ESRGAN Hybrid Only")
    parser.add_argument('--video' , '-v', required=True)
    parser.add_argument('--audio' , '-a', required=True)
    parser.add_argument('--output', '-o', default='upscale.mp4')
    parser.add_argument('--limit', type=int, default=0, help='Limit frames for testing (0=All)')
    args = parser.parse_args()

    # --- 1. QTGMC 前処理 ---
    try:
        clip = core.ffms2.Source(source=args.video)
        clip = havsfunc.QTGMC(
            clip, 
            Preset='Slow', 
            TFF=True, 
            NoiseProcess=1, 
            EZDenoise=2.0, 
            Denoiser="dfttest", 
            ChromaNoise=True, 
            GrainRestore=0.4, 
            Sharpness=1.0
        )
        clip = core.resize.Bicubic(clip, width=640, height=480, format=vs.RGB24, matrix_in_s="709")
    except Exception as e:
        print(f"❌ VapourSynthエラー: {e}"); sys.exit(1)

    total_frames = args.limit if args.limit > 0 else clip.num_frames
    fps = clip.fps.numerator / clip.fps.denominator
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 2. AIモデル準備 (Real-ESRGANのみ) ---
    model_r = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    
    # 自然な質感重視
    ups_net = RealESRGANer(
        scale=4, 
        model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth', 
        model=model_r, tile=1024, half=True, device=device
    )
    # 輪郭強調重視
    ups_gan = RealESRGANer(
        scale=4, 
        model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth', 
        model=model_r, tile=1024, half=True, device=device
    )

    temp_v = 'temp_render.mp4'
    out = cv2.VideoWriter(temp_v, cv2.VideoWriter_fourcc(*'mp4v'), fps, (clip.width * 4, clip.height * 4))

    # --- 3. 実行ループ ---
    print(f"✅ 処理開始 (Upscale Only)")
    pbar = tqdm(total=total_frames)

    for i in range(total_frames):
        vs_f = clip.get_frame(i)
        img = cv2.cvtColor(np.dstack([np.array(vs_f[j]) for j in range(vs_f.format.num_planes)]), cv2.COLOR_RGB2BGR)
        
        # A. Upscale (Hybrid 70/30)
        # 処理を高速化したい場合は、片方のモデルのみに絞ることも可能です
        r_net, _ = ups_net.enhance(img, outscale=4)
        r_gan, _ = ups_gan.enhance(img, outscale=4)
        output = cv2.addWeighted(r_net, 0.6, r_gan, 0.4, 0)
        
        out.write(output)
        pbar.update(1)

        if i % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    out.release()
    pbar.close()

    if args.limit == 0:
        process_final_video(temp_v, args.audio, args.output)
        if os.path.exists(temp_v): os.remove(temp_v)
    else:
        print(f"\n⚠️ テスト完了: {temp_v} を確認")

    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*30)
    print(f"🏁 処理完了!")
    print(f"開始時刻: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"終了時刻: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"総処理時間: {duration}")
    print("="*30)


if __name__ == "__main__":
    main()
