// fracture_surface/frontend/src/firebase.js

import { initializeApp } from "firebase/app";

import {
  getFirestore,
  collection,
  addDoc,
  getDocs,
  deleteDoc,
  doc,
  orderBy,
  query,
  limit,
} from "firebase/firestore";


// ==========================================
// 0. DB 저장 On/Off 스위치
// ==========================================
// true  → Firestore 실제로 읽기/쓰기 수행
// false → 네트워크 호출 없이 즉시 반환 (로컬 개발용, 빠름)

const DB_STORE = true;


// ==========================================
// 1. Firebase 프로젝트 설정
// ==========================================

const firebaseConfig = {
  apiKey: "AIzaSyDUXgli766zJ_Iww8UKINX7tLbLkqFHiyE",
  authDomain: "grain-a489f.firebaseapp.com",
  projectId: "grain-a489f",
  storageBucket: "grain-a489f.firebasestorage.app",
  messagingSenderId: "604330631780",
  appId: "1:604330631780:web:fa3dc1fa808914e3a6b571",
  measurementId: "G-23LTZ2MHG6",
};

const app = initializeApp(firebaseConfig);
export const db = getFirestore(app);

const HISTORY_COLLECTION = "analysisHistory";


// ==========================================
// 2. Firestore 저장 데이터 변환 (기존과 동일)
// ==========================================

function sanitizeForFirestore(value) {
  if (value === undefined) return null;
  if (value === null) return null;

  if (Array.isArray(value)) {
    return value.map((item) => {
      if (Array.isArray(item)) {
        return {
          __nestedArray: true,
          values: sanitizeForFirestore(item),
        };
      }
      return sanitizeForFirestore(item);
    });
  }

  if (typeof value === "object") {
    const cleaned = {};
    Object.entries(value).forEach(([key, item]) => {
      cleaned[key] = sanitizeForFirestore(item);
    });
    return cleaned;
  }

  return value;
}


// ==========================================
// 3. Firestore 데이터 복원 (기존과 동일)
// ==========================================

function restoreFromFirestore(value) {
  if (Array.isArray(value)) {
    return value.map(restoreFromFirestore);
  }

  if (value && typeof value === "object") {
    if (value.__nestedArray === true) {
      return restoreFromFirestore(value.values);
    }

    const restored = {};
    Object.entries(value).forEach(([key, item]) => {
      restored[key] = restoreFromFirestore(item);
    });
    return restored;
  }

  return value;
}


// ==========================================
// 4. 분석 기록 저장
// ==========================================

export async function addHistoryItem(item) {

  if (!DB_STORE) {
    console.log("[DB_STORE=false] 저장 건너뜀");
    return null;
  }

  try {
    const cleanedItem = sanitizeForFirestore(item);

    const docRef = await addDoc(
      collection(db, HISTORY_COLLECTION),
      cleanedItem
    );

    console.log("Firestore 기록 저장 성공:", docRef.id);

    return docRef.id;

  } catch (error) {
    console.error("Firestore 기록 저장 실패:", error);
    throw error;
  }
}


// ==========================================
// 5. 최근 분석 기록 불러오기
// ==========================================

export async function fetchHistory() {

  if (!DB_STORE) {
    console.log("[DB_STORE=false] 불러오기 건너뜀");
    return [];
  }

  try {
    const q = query(
      collection(db, HISTORY_COLLECTION),
      orderBy("id", "desc"),
      limit(20)
    );

    const snapshot = await getDocs(q);

    const items = snapshot.docs.map((document) => {
      const data = restoreFromFirestore(document.data());
      return { ...data, docId: document.id };
    });

    console.log("Firestore 기록 불러오기 성공:", items.length);

    return items;

  } catch (error) {
    console.error("Firestore 기록 불러오기 실패:", error);
    throw error;
  }
}


// ==========================================
// 6. 분석 기록 전체 삭제
// ==========================================

export async function clearAllHistory() {

  if (!DB_STORE) {
    console.log("[DB_STORE=false] 삭제 건너뜀");
    return;
  }

  try {
    const snapshot = await getDocs(collection(db, HISTORY_COLLECTION));

    await Promise.all(
      snapshot.docs.map((document) =>
        deleteDoc(doc(db, HISTORY_COLLECTION, document.id))
      )
    );

    console.log("Firestore 전체 기록 삭제 완료");

  } catch (error) {
    console.error("Firestore 기록 삭제 실패:", error);
    throw error;
  }
}