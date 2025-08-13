from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import re
import json
from pathlib import Path
import uuid
import requests
from werkzeug.utils import secure_filename
from datetime import datetime
import sys
import subprocess
import torch
import gc
import random

# instruct-pix2pix 모듈 import를 위한 경로 추가
sys.path.append('./instruct-pix2pix')

app = Flask(__name__)

# 포트 설정
PORT = 5000

# CORS 설정 (Next.js에서 접근 허용)
CORS(app, origins=["http://localhost:3000"])

# 폴더 설정
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# 폴더 생성
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# 전역 변수로 모델 저장
model = None
model_wrap = None
model_wrap_cfg = None
null_token = None
device = None

AXIS_MAX_X = 400
AXIS_MAX_Y = 1100

def parse_contours(raw_text):
    blocks = re.split(r'#\s*contour\s*\d+', raw_text)[1:]
    contours = []
    for blk in blocks:
        pts = []
        for line in blk.strip().splitlines():
            m = re.match(r'\s*(\d+)\s*,\s*(\d+)\s*$', line)
            if m:
                pts.append((int(m.group(1)), int(m.group(2))))
        if pts:
            contours.append(pts)
    return contours

def scale_contours(contours):
    xs = [x for c in contours for x,_ in c]
    ys = [y for c in contours for _,y in c]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    def mx(px): return (px - minx) / (maxx - minx) * AXIS_MAX_X
    def my(py): return (py - miny) / (maxy - miny) * AXIS_MAX_Y

    return [
        [(mx(x), my(y)) for x,y in contour]
        for contour in contours
    ]
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model_once():
    """모델을 한 번만 로드하는 함수"""
    global model, model_wrap, model_wrap_cfg, null_token, device
    
    if model is not None:
        return  # 이미 로드되어 있으면 리턴
    
    print("[+] Loading model...")
    
    # GPU 메모리 정리 및 최적화 설정
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        # 메모리 할당 최적화 설정
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64,roundup_power2_divisions:16'
    
    # instruct-pix2pix 디렉토리로 이동
    original_dir = os.getcwd()
    instruct_dir = os.path.join(original_dir, '..', 'instruct-pix2pix')
    
    # Python 경로에 instruct-pix2pix와 stable_diffusion 추가
    if instruct_dir not in sys.path:
        sys.path.insert(0, instruct_dir)
    stable_diffusion_path = os.path.join(instruct_dir, 'stable_diffusion')
    if stable_diffusion_path not in sys.path:
        sys.path.insert(0, stable_diffusion_path)
    
    os.chdir(instruct_dir)
    
    try:
        from omegaconf import OmegaConf
        from stable_diffusion.ldm.util import instantiate_from_config
        import k_diffusion as K
        from einops import rearrange
        import math
        from PIL import Image, ImageOps
        import numpy as np
        import cv2
        import mediapipe as mp
        
        # 모델 로드
        config = OmegaConf.load("configs/generate.yaml")
        model = instantiate_from_config(config.model)
        
        # 체크포인트 로드
        pl_sd = torch.load("checkpoints/instruct-pix2pix-00-22000.ckpt", map_location="cpu")
        sd = pl_sd["state_dict"]
        model.load_state_dict(sd, strict=False)
        
        # GPU 사용
        model.eval().cuda()
        device = "cuda"
        print("[+] Using CUDA")
        
        model_wrap = K.external.CompVisDenoiser(model)
        model_wrap_cfg = CFGDenoiser(model_wrap)
        null_token = model.get_learned_conditioning([""])
        
        print("[+] Model loaded successfully")
        
    except Exception as e:
        print(f"[!] Error loading model: {e}")
        raise e
    finally:
        # 원래 디렉토리로 복귀
        os.chdir(original_dir)

class CFGDenoiser(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = torch.cat([z, z, z], dim=0)
        cfg_sigma = torch.cat([sigma, sigma, sigma], dim=0)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)

def process_image_with_model(input_path, output_path, instruction="Remove the background and produce a black line-art tracing that clearly defines the contours of the person’s body, hands, shoulders, neck, and face. Exaggerate the eyes, nose, and mouth, and do not apply any color fill. Trace the contours as closely as possible to the original image."):
    from PIL import Image, ImageOps
    import numpy as np
    import cv2
    import mediapipe as mp
    import math
    from einops import rearrange
    import k_diffusion as K
    global model, model_wrap_cfg, null_token, device
    
    if model is None:
        raise Exception("Model not loaded")
    
    # GPU 메모리 정리
    torch.cuda.empty_cache()
    gc.collect()
    
    # 이미지 로드 및 전처리
    input_image = Image.open(input_path).convert("RGB")
    
    # PIL → OpenCV (배경 제거)
    image_np = np.array(input_image)[:, :, ::-1].copy()
    mp_selfie = mp.solutions.selfie_segmentation
    segment = mp_selfie.SelfieSegmentation(model_selection=1)
    result = segment.process(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    condition = result.segmentation_mask > 0.6
    bg = np.ones(image_np.shape, dtype=np.uint8) * 255
    output_np = np.where(condition[..., None], image_np, bg)
    input_image = Image.fromarray(output_np[:, :, ::-1])
    
    # 크기 조정
    width, height = input_image.size
    factor = 256 / max(width, height)  # 512에서 256으로 변경
    factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
    width = int((width * factor) // 64) * 64
    height = int((height * factor) // 64) * 64
    input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)
    
    # 모델 추론
    with torch.no_grad(), torch.autocast("cuda"):
        # 배치 크기를 1로 명시적 설정
        torch.cuda.set_per_process_memory_fraction(0.8)  # GPU 메모리의 80%만 사용
        
        cond = {
            "c_crossattn": [model.get_learned_conditioning([instruction])],
            "c_concat": [model.encode_first_stage(
                2 * torch.tensor(np.array(input_image)).float().div(255).sub(0.5).mul(2).permute(2, 0, 1).unsqueeze(0).to(model.device)
            ).mode()]
        }
        
        uncond = {
            "c_crossattn": [null_token],
            "c_concat": [torch.zeros_like(cond["c_concat"][0])],
        }
        
        sigmas = model_wrap.get_sigmas(100) 
        extra_args = {
            "cond": cond,
            "uncond": uncond,
            "text_cfg_scale": 7.5,
            "image_cfg_scale": 1.5,
        }
        
        torch.manual_seed(random.randint(0, 100000))  # 랜덤 seed
        z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
        z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)
        
        x = model.decode_first_stage(z)
        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
        x = 255.0 * rearrange(x, "1 c h w -> h w c")
        edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())
        
        edited_image.save(output_path)
        
        # 윤곽선 추출
        edited_np = np.array(edited_image.convert("RGB"))
        gray = cv2.cvtColor(edited_np, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, threshold1=50, threshold2=150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 좌표 txt 파일 저장
        contours_filename = f"contours_{uuid.uuid4()}.txt"
        contours_path = os.path.join(app.config['RESULT_FOLDER'], contours_filename)
        with open(contours_path, "w") as f:
            for i, contour in enumerate(contours):
                f.write(f"# contour {i}\n")
                for point in contour:
                    x, y = point[0]
                    f.write(f"{x},{y}\n")
                f.write("\n")

        # 윤곽선 이미지 저장
        outline_filename = f"outline_{uuid.uuid4()}.png"
        outline_path = os.path.join(app.config['RESULT_FOLDER'], outline_filename)
        inverted = cv2.bitwise_not(edges)
        outline_image = Image.fromarray(inverted)
        outline_image.save(outline_path)
        
        # GPU 메모리 정리
        del z, x, cond, uncond, sigmas, extra_args
        torch.cuda.empty_cache()
        gc.collect()
        
        # 추가 메모리 정리
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': '파일이 없습니다.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': '파일이 선택되지 않았습니다.'}), 400
    
    if file and allowed_file(file.filename):
        # 고유한 파일명 생성
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # 파일 저장
        file.save(filepath)
        
        # 이미지 처리
        try:
            # 결과 파일명 생성
            result_filename = f"result_{uuid.uuid4()}.png"
            result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
            
            # 로드된 모델로 이미지 처리
            process_image_with_model(filepath, result_path, "Remove the background and produce a black line-art tracing that clearly defines the contours of the person’s body, hands, shoulders, neck, and face. Exaggerate the eyes, nose, and mouth, and do not apply any color fill. Trace the contours as closely as possible to the original image.")
            
            return jsonify({
                'success': True,
                'uploaded_file': unique_filename,
                'result_file': result_filename,
                'message': '이미지가 성공적으로 처리되었습니다.'
            }), 200
                
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'이미지 처리 중 오류가 발생했습니다: {str(e)}'
            }), 500
    
    return jsonify({'success': False, 'error': '지원하지 않는 파일 형식입니다.'}), 400

@app.route('/api/generate', methods=['POST'])
def generate_new_image():
    data = request.get_json()
    uploaded_file = data.get('uploaded_file')
    
    if not uploaded_file:
        return jsonify({'success': False, 'error': '업로드된 파일 정보가 없습니다.'}), 400
    
    try:
        # 원본 파일 경로
        original_filepath = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file)
        
        if not os.path.exists(original_filepath):
            return jsonify({'success': False, 'error': '원본 파일을 찾을 수 없습니다.'}), 400
        
        # 새로운 결과 파일명 생성
        result_filename = f"result_{uuid.uuid4()}.png"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        
        # 로드된 모델로 이미지 처리 (랜덤 시드)
        process_image_with_model(original_filepath, result_path, "Remove the background and produce a black line-art tracing that clearly defines the contours of the person’s body, hands, shoulders, neck, and face. Exaggerate the eyes, nose, and mouth, and do not apply any color fill. Trace the contours as closely as possible to the original image.")
        
        return jsonify({
            'success': True,
            'result_file': result_filename,
            'message': '새로운 이미지가 생성되었습니다.'
        }), 200
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'이미지 처리 중 오류가 발생했습니다: {str(e)}'
        }), 500
@app.route('/')
def index():
    return "API is working!"

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'Flask server is running',
        'port': PORT
    }), 200

@app.errorhandler(413)
def too_large(e):
    return jsonify({'success': False, 'error': '파일 크기가 너무 큽니다. 5MB 이하의 파일을 업로드해주세요.'}), 413

if __name__ == '__main__':
    # 서버 시작 시 모델 로드
    load_model_once()
    app.run(debug=True, host='0.0.0.0', port=PORT,use_reloader=False)