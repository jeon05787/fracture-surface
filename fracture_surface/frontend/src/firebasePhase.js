import {
  collection,
  addDoc,
  getDocs,
  deleteDoc,
  updateDoc,
  doc,
  orderBy,
  query,
  limit,
} from "firebase/firestore";
import { db } from "./firebase"; // 기존 프로젝트의 db 재사용

const PHASE_COLLECTION = "phaseHistory";
const MAX_HISTORY = 20;
const MAX_DOC_BYTES = 900_000; // Firestore 한도 1MiB 보다 여유 있게

// 이미지(dataURL)를 축소해 JPEG dataURL로 반환
export function resizeDataUrl(dataUrl, maxSize = 480, quality = 0.6) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const scale = Math.min(1, maxSize / Math.max(img.width, img.height));
      const canvas = document.createElement("canvas");
      canvas.width = Math.round(img.width * scale);
      canvas.height = Math.round(img.height * scale);
      canvas.getContext("2d").drawImage(img, 0, 0, canvas.width, canvas.height);
      resolve(canvas.toDataURL("image/jpeg", quality));
    };
    img.onerror = reject;
    img.src = dataUrl;
  });
}

// 문서 크기 대략 계산 (UTF-8 기준)
function estimateBytes(obj) {
  return new Blob([JSON.stringify(obj)]).size;
}

// 저장용 기록 객체 생성 (가벼운 정보만)
export async function buildPhaseHistoryItem({ title, originalDataUrl, data }) {
  const thumbnail = await resizeDataUrl(originalDataUrl, 180, 0.65);
  const overlaySmall = data.overlay_image
    ? await resizeDataUrl(data.overlay_image, 480, 0.6)
    : null;

  const item = {
    id: Date.now(),
    title,
    time: new Date().toLocaleString("ko-KR"),
    image: thumbnail,
    result: {
      status: "success",
      image_width: data.image_width,
      image_height: data.image_height,
      phase_distribution: data.phase_distribution,
      overlay_image: overlaySmall, // 축소본 (복원용)
      overlay_is_preview: true,
    },
  };

  // 크기 초과 시 overlay 제거
  if (estimateBytes(item) > MAX_DOC_BYTES) {
    item.result.overlay_image = null;
  }

  return item;
}

export async function addPhaseHistoryItem(item) {
  const ref = await addDoc(collection(db, PHASE_COLLECTION), item);
  await trimOldHistory(); // 20개 초과분 삭제
  return ref.id;
}

export async function fetchPhaseHistory() {
  const q = query(
    collection(db, PHASE_COLLECTION),
    orderBy("id", "desc"),
    limit(MAX_HISTORY)
  );
  const snap = await getDocs(q);
  return snap.docs.map((d) => ({ ...d.data(), docId: d.id }));
}

export async function renamePhaseHistoryItem(docId, title) {
  await updateDoc(doc(db, PHASE_COLLECTION, docId), { title });
}

export async function clearAllPhaseHistory() {
  const snap = await getDocs(collection(db, PHASE_COLLECTION));
  await Promise.all(
    snap.docs.map((d) => deleteDoc(doc(db, PHASE_COLLECTION, d.id)))
  );
}

// 오래된 기록 정리
async function trimOldHistory() {
  const q = query(collection(db, PHASE_COLLECTION), orderBy("id", "desc"));
  const snap = await getDocs(q);
  const extra = snap.docs.slice(MAX_HISTORY);
  await Promise.all(extra.map((d) => deleteDoc(doc(db, PHASE_COLLECTION, d.id))));
}