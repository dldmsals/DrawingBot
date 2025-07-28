from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import uuid
from werkzeug.utils import secure_filename
from datetime import datetime

app = Flask(__name__)

# 포트 설정
PORT = 5001

# CORS 설정 (Next.js에서 접근 허용)
CORS(app, origins=["http://localhost:3000"])

# 업로드 설정
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# uploads 폴더 생성
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        # 파일이 요청에 포함되어 있는지 확인
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': '파일이 없습니다.'
            }), 400

        file = request.files['file']

        # 파일이 선택되었는지 확인
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': '파일이 선택되지 않았습니다.'
            }), 400

        # 파일 확장자 검증
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': '이미지 파일만 업로드 가능합니다.'
            }), 400

        if file:
            # 안전한 파일명 생성
            original_filename = secure_filename(file.filename)
            file_extension = original_filename.rsplit('.', 1)[1].lower()
            
            # 고유한 파일명 생성 (타임스탬프 + UUID)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_id = str(uuid.uuid4())[:8]
            filename = f"{timestamp}_{unique_id}.{file_extension}"
            
            # 파일 저장
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # 파일 URL 생성 (포트 번호 일치)
            file_url = f"http://localhost:{PORT}/uploads/{filename}"

            return jsonify({
                'success': True,
                'url': file_url,
                'filename': filename
            }), 200

    except Exception as e:
        print(f"Upload error: {str(e)}")
        return jsonify({
            'success': False,
            'error': '서버 오류가 발생했습니다.'
        }), 500

# 업로드된 파일 서빙
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# 서버 상태 확인용 엔드포인트
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'Flask server is running',
        'port': PORT
    }), 200

# 에러 핸들러
@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'success': False,
        'error': '파일 크기가 너무 큽니다. 5MB 이하의 파일을 업로드해주세요.'
    }), 413

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=PORT)