"use client"

import React, { useState, useRef } from 'react';
import Image from 'next/image';
import styles from './PhotoUpload.module.css';

interface UploadResponse {
  success: boolean;
  uploaded_file?: string;
  result_file?: string;
  message?: string;
  error?: string;
}

const PhotoUpload: React.FC = () => {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [resultImage, setResultImage] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // 파일 유효성 검사
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp'];
    if (!allowedTypes.includes(file.type)) {
      setError('지원하지 않는 파일 형식입니다. JPEG, PNG, GIF, WebP 파일만 업로드 가능합니다.');
      return;
    }

    if (file.size > 5 * 1024 * 1024) {
      setError('파일 크기가 너무 큽니다. 5MB 이하의 파일을 업로드해주세요.');
      return;
    }

    // 이미지 미리보기
    const reader = new FileReader();
    reader.onload = (e) => {
      setUploadedImage(e.target?.result as string);
    };
    reader.readAsDataURL(file);

    // 백엔드로 파일 전송
    setIsProcessing(true);
    setError(null);
    setResultImage(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:5000/api/upload', {
        method: 'POST',
        body: formData,
      });

      const data: UploadResponse = await response.json();
      console.log("post 성공");

      if (data.success) {
        // 결과 이미지 URL 생성
        const resultImageUrl = `http://localhost:5000/results/${data.result_file}`;
        setResultImage(resultImageUrl);
        setUploadedFileName(data.uploaded_file || null);
      } else {
        setError(data.error || '이미지 처리 중 오류가 발생했습니다.');
      }
    } catch (err) {
      setError('서버와의 연결에 실패했습니다.1111');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleGenerate = async () => {
    if (!uploadedFileName) {
      setError('먼저 이미지를 업로드해주세요.');
      return;
    }

    setIsProcessing(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:5000/api/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          uploaded_file: uploadedFileName
        }),
      });

      const data = await response.json();

      if (data.success) {
        // 새로운 결과 이미지 URL 생성
        const resultImageUrl = `http://localhost:5000/results/${data.result_file}`;
        setResultImage(resultImageUrl);
      } else {
        setError(data.error || '이미지 생성 중 오류가 발생했습니다.');
      }
    } catch (err) {
      setError('서버와의 연결에 실패했습니다.22222');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragOver(false);
    const files = event.dataTransfer.files;
    if (files.length > 0) {
      const file = files[0];
      if (fileInputRef.current) {
        fileInputRef.current.files = files;
        handleFileChange({ target: { files } } as React.ChangeEvent<HTMLInputElement>);
      }
    }
  };

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragOver(false);
  };

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className={styles.container}>
      {/* 왼쪽: 업로드 영역 */}
      <div className={styles.left}>
        <div 
          className={`${styles.dropArea} ${isDragOver ? styles.hover : ''}`} 
          onClick={handleClick}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="hidden"
          />
          
          {uploadedImage ? (
            <div className={styles.uploadedImageContainer}>
              <img 
                src={uploadedImage} 
                alt="Uploaded image" 
                className={styles.uploadedImage}
              />
              <p className={styles.uploadText}>
                클릭하여 다른 이미지 선택
              </p>
            </div>
          ) : (
            <div className={styles.uploadContent}>
              <div className={styles.uploadIcon}>
                <svg className={styles.icon} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
              </div>
              <div className={styles.uploadText}>
                <p className={styles.uploadTitle}>
                  이미지를 드래그하거나 클릭하여 업로드
                </p>
                <p className={styles.uploadSubtitle}>
                  JPEG, PNG, GIF, WebP (최대 5MB)
                </p>
              </div>
            </div>
          )}
        </div>

        {isProcessing && (
          <div className={styles.processing}>
            <div className={styles.spinner}></div>
            <span>이미지 처리 중...</span>
          </div>
        )}

        {error && (
          <div className={styles.error}>
            <p>{error}</p>
          </div>
        )}
      </div>

      {/* 오른쪽: 결과 영역 */}
      <div className={styles.right}>
        <h2 className={styles.resultTitle}>변환 결과</h2>
        
        {resultImage && (
          <div className={styles.generateButtonContainer}>
            <button 
              className={styles.generateButton}
              onClick={handleGenerate}
              disabled={isProcessing}
            >
              {isProcessing ? '생성 중...' : 'Generate'}
            </button>
          </div>
        )}
        
        <div className={styles.resultArea}>
          {resultImage ? (
            <div className={styles.resultContainer}>
              <img 
                src={resultImage} 
                alt="Result image" 
                className={styles.resultImage}
              />
              <p className={styles.resultText}>
                AI가 생성한 이미지
              </p>
            </div>
          ) : (
            <div className={styles.resultPlaceholder}>
              <div className={styles.placeholderIcon}>
                <svg className={styles.icon} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
              </div>
              <p className={styles.placeholderText}>
                이미지를 업로드하면 여기에 결과가 표시됩니다
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PhotoUpload;
