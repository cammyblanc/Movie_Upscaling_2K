# Movie_Upscaling_2K
upscaling analog movies to 2K, using realESRgan


# ライブラリとして使っているファイル名。必ず必要
 - havsfunc.py
 - mvsfunc.py
 - adjust.py


# 通常

 - uv run image_qual_v4.py -v test1.mpg -a test1.mpg -o test_gan.mp4
 - uv run image_qual_v4_cf.py -v test1.mpg -a test1.mpg -o test_gan_cf07.mp4 --w 0.7
