export default function SimilarImageCard({ item }) {
  return (
    <div className="bg-white rounded-2xl border p-4 shadow-sm">
      <div className="h-48 rounded-xl bg-slate-100 overflow-hidden border mb-4 flex items-center justify-center">
        {item.image ? (
          <img
            src={item.image}
            alt="유사 이미지"
            className="w-full h-full object-cover"
          />
        ) : (
          <p className="text-sm text-slate-400">No Image</p>
        )}
      </div>

      <div className="space-y-2">
        <p className="text-sm text-slate-500">예측 유형</p>
        <p className="font-bold text-slate-900">{item.type}</p>

        <p className="text-sm text-slate-500 mt-3">유사도</p>
        <p className="font-bold text-blue-600">{item.similarity}</p>
      </div>
    </div>
  );
}