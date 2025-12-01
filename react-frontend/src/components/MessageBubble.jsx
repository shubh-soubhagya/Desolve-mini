export default function MessageBubble({ role, text }) {
  const mine = role === "user";

  return (
    <div className={`flex ${mine ? "justify-end" : "justify-start"} mb-3`}>
      <div className={`px-4 py-2 rounded-lg max-w-lg 
        ${mine ? "bg-blue-600 text-white" : "bg-gray-200"}`}>
        {text}
      </div>
    </div>
  );
}
