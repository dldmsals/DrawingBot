import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: '사진 업로드',
  description: '사진 업로드 앱',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ko">
      <body style={{ margin: 0, padding: 0, fontFamily: 'sans-serif' }}>
        {children}
      </body>
    </html>
  );
}