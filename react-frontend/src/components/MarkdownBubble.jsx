import marked from "../markdownConfig";

export default function MarkdownBubble({ text, mine }) {
  return (
    <div className={`flex ${mine ? "justify-end" : "justify-start"} mb-3`}>
      <div
        className={`px-4 py-3 rounded-lg max-w-xl whitespace-pre-wrap ${
          mine ? "bg-blue-600 text-white" : "bg-gray-100 text-gray-900"
        }`}
        dangerouslySetInnerHTML={{
          __html: mine ? text : marked.parse(text)
        }}
      />
    </div>
  );
}
