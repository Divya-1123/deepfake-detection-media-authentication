export const IMAGE_EXTS = ['.jpg', '.jpeg', '.png', '.webp'];
export const VIDEO_EXTS = ['.mp4', '.avi', '.mov', '.mkv'];
export const MAX_IMAGE_MB = 10;
export const MAX_VIDEO_MB = 100;

const ext = (name) => name.slice(name.lastIndexOf('.')).toLowerCase();
export const isVideoFile = (file) => !!file && VIDEO_EXTS.includes(ext(file.name));

/** Returns an error message, or '' when the file is acceptable. Mirrors the backend limits. */
export function validateFile(file) {
  const e = ext(file.name);
  if (![...IMAGE_EXTS, ...VIDEO_EXTS].includes(e)) return `Unsupported file type "${e || file.name}". Use JPG, PNG, WEBP, MP4, AVI, MOV or MKV.`;
  const limit = VIDEO_EXTS.includes(e) ? MAX_VIDEO_MB : MAX_IMAGE_MB;
  if (file.size === 0) return 'This file is empty.';
  if (file.size > limit * 1024 * 1024) return `File is too large. Maximum is ${limit} MB for ${VIDEO_EXTS.includes(e) ? 'videos' : 'images'}.`;
  return '';
}
