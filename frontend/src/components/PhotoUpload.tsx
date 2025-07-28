import React, { useState, useRef, DragEvent, ChangeEvent } from 'react';
import styles from './PhotoUpload.module.css';

interface UploadResponse {
  success: boolean;
  url?: string;
  error?: string;
}

const PhotoUpload: React.FC = () => {
  const [uploadedImageUrl, setUploadedImageUrl] = useState<string | null>(null);
  const [isHovering, setIsHovering] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsHovering(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsHovering(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsHovering(false);
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileUpload(files[0]);
    }
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFileUpload(files[0]);
    }
  };

  const handleFileUpload = async (file: File) => {
    if (!file.type.startsWith('image/')) {
      alert('이미지 파일만 업로드 가능합니다.');
      return;
    }

    setIsUploading(true);
    
    const formData = new FormData();
    formData.append('file', file);

    try {
      // Flask 백엔드 서버로 요청 (포트 5001로 변경)
      const response = await fetch('http://localhost:5001/api/upload', {
        method: 'POST',
        body: formData,
      });

      const data: UploadResponse = await response.json();
      
      console.log('응답 받은 데이터:', data);
      
      if (data.success && data.url) {
        setUploadedImageUrl(data.url);
      } else {
        alert('업로드 실패: ' + (data.error || '알 수 없는 오류'));
      }
    } catch (error) {
      console.error('업로드 중 오류:', error);
      alert('업로드 중 오류가 발생했습니다.');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className={styles.container}>
      {/* 왼쪽: 드래그 앤 드롭 업로드 박스 */}
      <div className={styles.left}>
        <div 
          className={`${styles.dropArea} ${isHovering ? styles.hover : ''}`}
          onClick={handleClick}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          {isUploading ? (
            <p>📤 업로드 중...</p>
          ) : uploadedImageUrl ? (
            <img 
              src={uploadedImageUrl} 
              alt="Uploaded" 
              className={styles.uploadedImage}
            />
          ) : (
            <p>📤 사진을 드래그하거나 클릭하세요</p>
          )}
          <input
            type="file"
            ref={fileInputRef}
            accept="image/*"
            onChange={handleFileChange}
            style={{ display: 'none' }}
          />
        </div>
      </div>

      {/* 오른쪽: 비어있음 */}
      <div className={styles.right}>
        {/* 여기엔 아직 아무것도 없음 */}
      </div>
    </div>
  );
};

export default PhotoUpload;