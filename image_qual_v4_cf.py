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

# --- 書き換え後のコード ---
try:
    from codeformer.codeformer_arch import CodeFormer
    from facexlib.utils.face_restoration_helper import FaceRestoreHelper
    print("✅ ライブラリの読み込みに成功しました。")
except ImportError as e:
    print(f"❌ インポートエラーの詳細: {e}")
    # 具体的に何が足りないかを表示する
    import traceback
    traceback.print_exc()
    sys.exit(1)

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
      #  "unsharp=7:7:0.5:7:7:0.0,"
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

def color_transfer(source, target):
    """
    輝度（ディテール）はAIのものを使い、色合いは元の画像に戻すことで、
    四角い境界線や、髪の毛に混じる青み・白飛びを防止する。
    """
    # 画像を LAB空間（L:明るさ, a/b:色合い）に変換
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

    # L（輝度＝ディテール）: AIの解像度を活かしつつ、白飛びを抑えるため元画像を20%混ぜる
    target_lab[:, :, 0] = target_lab[:, :, 0] * 0.15 + source_lab[:, :, 0] * 0.85
    
    # a, b（色合い）: AIの勝手な色変更（青いシミなど）を消すため、元画像の色を90%適用する
    target_lab[:, :, 1] = target_lab[:, :, 1] * 0.001 + source_lab[:, :, 1] * 0.999
    target_lab[:, :, 2] = target_lab[:, :, 2] * 0.001 + source_lab[:, :, 2] * 0.999

    target_lab = np.clip(target_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(target_lab, cv2.COLOR_LAB2BGR)

# def color_transfer(source, target):
#     """
#     source (元の顔) の色調を target (AI復元後の顔) に転送する
#     """
#     source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
#     target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
# 
#     for i in range(3):
#         mu_s, std_s = source[:, :, i].mean(), source[:, :, i].std()
#         mu_t, std_t = target[:, :, i].mean(), target[:, :, i].std()
#         
#         # 0除算防止
#         if std_t < 1e-6: continue
#             
#         target[:, :, i] = (target[:, :, i] - mu_t) * (std_s / std_t) + mu_s
# 
#     target = np.clip(target, 0, 255).astype(np.uint8)
#     return cv2.cvtColor(target, cv2.COLOR_LAB2BGR)

def main():
    start_time = datetime.now()
    parser = argparse.ArgumentParser(description="LD Restoration: Hybrid AI + CodeFormer")
    parser.add_argument('--video' , '-v', required=True)
    parser.add_argument('--audio' , '-a', required=True)
    parser.add_argument('--output', '-o', default='restored_2k_cf.mp4')
    parser.add_argument('--w', type=float, default=0.7, help='Fidelity: 0=Sharp(AI), 1=Original')
    parser.add_argument('--limit', type=int, default=0, help='Limit frames for testing (0=All)')
    args = parser.parse_args()

    # --- 1. QTGMC 前処理 (EZDenoise有効) ---
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

    # --- 2. AIモデル準備 (RTX 3080 Ti 16GB 最適化) ---
    # Real-ESRGAN
    model_r = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    # 自然な質感重視の Net
    ups_net = RealESRGANer(
        scale=4, 
        model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth', 
        model=model_r, tile=1024, half=True, device=device
    )
    # 輪郭強調重視の GAN
    ups_gan = RealESRGANer(
        scale=4, 
        model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth', 
        model=model_r, tile=1024, half=True, device=device
    )

    # CodeFormer (顔・髪復元)
    face_helper = FaceRestoreHelper(1, face_size=512, det_model='retinaface_resnet50', device=device)
    cf_net = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, connect_list=['32', '64', '128', '256']).to(device)
    
    ckpt = 'weights/CodeFormer/codeformer.pth'
    if os.path.exists(ckpt):
        cf_net.load_state_dict(torch.load(ckpt, map_location='cpu')['params_ema'])
    cf_net.eval()

    temp_v = 'temp_render.mp4'
    out = cv2.VideoWriter(temp_v, cv2.VideoWriter_fourcc(*'mp4v'), fps, (clip.width * 4, clip.height * 4))

    # --- 3. 実行ループ ---
    print(f"✅ 処理開始: w={args.w}")
    pbar = tqdm(total=total_frames)

    for i in range(total_frames):
        vs_f = clip.get_frame(i)
        img = cv2.cvtColor(np.dstack([np.array(vs_f[j]) for j in range(vs_f.format.num_planes)]), cv2.COLOR_RGB2BGR)
        
        # A. Upscale (Hybrid 40/60)
        r_net, _ = ups_net.enhance(img, outscale=4)
        r_gan, _ = ups_gan.enhance(img, outscale=4)
        output = cv2.addWeighted(r_net, 0.7, r_gan, 0.3, 0)
        
        # B. 顔と髪の復元 (CodeFormer)
        face_helper.clean_all()
        face_helper.read_image(output)
        face_helper.get_face_landmarks_5(only_center_face=False, eye_dist_threshold=10)
        face_helper.align_warp_face()

# --- 修正: ガウスマスクを使ってAIの影響を顔の中心だけに限定する ---
        for f in face_helper.cropped_faces:
            # 1. BGR から RGB に変換
            f_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            
            # 2. [-1.0, 1.0] に正規化してテンソル化
            t_f = torch.from_numpy(f_rgb).permute(2, 0, 1).float()
            t_f = (t_f / 255.0 - 0.5) / 0.5  
            t_f = t_f.unsqueeze(0).to(device)
            
            with torch.no_grad():
                # adain=True に戻す（顔自体の自然な色合わせはAIに任せます）
                out_f_t, *others = cf_net(t_f, w=args.w, adain=False)
                
                # 3. AIの出力 [-1.0, 1.0] を [0, 255] に戻す
                out_f_np = out_f_t.squeeze().permute(1, 2, 0).cpu().numpy()
                out_f_rgb = ((out_f_np * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)

            # 4. RGB から BGR に戻す
            out_f = cv2.cvtColor(out_f_rgb, cv2.COLOR_RGB2BGR)

            try:
                out_f = color_transfer(f, out_f)
            except Exception:
                pass # 万が一エラーが起きても停止させない

            # 5. 【超重要】円形のソフトマスクを作成し、境界線を完全に消す
            h, w, _ = f.shape
            mask = np.zeros((h, w), dtype=np.float32)
            
            # 中心の顔部分(半径の約35%)だけを白(1.0)に塗る
            radius = int(min(h, w) * 0.35)
            cv2.circle(mask, (w//2, h//2), radius, 1.0, -1)
            
            # 境界を強烈にボカす(グラデーションを作る)
            blur_size = int(min(h, w) * 0.2) | 1 # cv2のカーネルサイズは奇数必須のため調整
            mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
            
            # maskを (h, w, 1) の形に拡張（色計算用）
            mask = mask[..., np.newaxis]
            
            # 6. AI画像(out_f)と元画像(f)をマスクで合成
            # 中心はAIの顔、端にいくにつれて元画像(RealESRGAN)に100%戻るため四角い枠が出ない
            out_f = (out_f.astype(np.float32) * mask + f.astype(np.float32) * (1.0 - mask)).astype(np.uint8)

            face_helper.add_restored_face(out_f)
            
        face_helper.get_inverse_affine(None)
        # 復元した顔を元の2K画像にマージ
        output = face_helper.paste_faces_to_input_image()        
        out.write(output)
        pbar.update(1)
        # 30フレームごとに VRAM と RAM を徹底的に掃除する
        if i % 30 == 0:
            torch.cuda.empty_cache()  # GPUの掃除
            gc.collect()              # RAMの掃除

    out.release()
    pbar.close()
    if args.limit == 0: # 全体処理の場合のみ最終統合
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
